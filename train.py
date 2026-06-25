import config
from generator import Generator
from discriminator import Discriminator
from dataloader import getDataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import GradScaler, autocast
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch import ones_like, zeros_like
import torch
import os
from torchvision.utils import save_image
from kornia.color import lab_to_rgb
import time
import json
from contextlib import nullcontext


def trainStep(l, ab, gen, disc, discOpt, genOpt, genScaler, discScaler, bce, l1):
    l = l.to(config.device)
    ab = ab.to(config.device)

    use_amp = (config.device == "cuda")
    amp_ctx = autocast(config.device, torch.float16) if use_amp else nullcontext()

    with amp_ctx:
        generatedAB = gen.forward(l)
        discReal = disc.forward(l, ab)
        discGenerated = disc.forward(l, generatedAB.detach())
        discRealLoss = bce.forward(discReal, ones_like(discReal))
        discGeneratedLoss = bce.forward(discGenerated, zeros_like(discGenerated))
        discLoss = (discRealLoss + discGeneratedLoss) / 2

    disc.zero_grad()
    discOpt.zero_grad()
    if use_amp:
        discScaler.scale(discLoss).backward()
        discScaler.step(discOpt)
        discScaler.update()
    else:
        discLoss.backward()
        discOpt.step()

    with amp_ctx:
        discPredictions = disc.forward(l, generatedAB)
        genFakeLoss = bce.forward(discPredictions, ones_like(discPredictions))
        l1Loss = l1.forward(generatedAB, ab) * config.l1Lambda
        genLoss = genFakeLoss + l1Loss

    gen.zero_grad()
    genOpt.zero_grad()
    if use_amp:
        genScaler.scale(genLoss).backward()
        genScaler.step(genOpt)
        genScaler.update()
    else:
        genLoss.backward()
        genOpt.step()

    return genLoss.item(), discLoss.item()


def save_checkpoint(gen, disc, genOpt, discOpt, genScaler, discScaler,
                    epoch, history, bestGenLoss, silent=False):

    state = {
        "epoch": epoch,
        "gen_state_dict": gen.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "genOpt_state_dict": genOpt.state_dict(),
        "discOpt_state_dict": discOpt.state_dict(),
        "genScaler_state_dict": genScaler.state_dict(),
        "discScaler_state_dict": discScaler.state_dict(),
        "history": history,
        "bestGenLoss": bestGenLoss,
    }
    torch.save(state, config.resumeFile)
    if not silent:
        print(f"  >> 训练状态已保存到 {config.resumeFile}")


def load_checkpoint(gen, disc, genOpt, discOpt, genScaler, discScaler):
    """加载训练状态。返回 (epoch, history, bestGenLoss)。"""
    state = torch.load(config.resumeFile, map_location=config.device)
    gen.load_state_dict(state["gen_state_dict"])
    disc.load_state_dict(state["disc_state_dict"])
    genOpt.load_state_dict(state["genOpt_state_dict"])
    discOpt.load_state_dict(state["discOpt_state_dict"])
    genScaler.load_state_dict(state["genScaler_state_dict"])
    discScaler.load_state_dict(state["discScaler_state_dict"])
    # 兼容旧存档（可能没有 bestGenLoss 字段）
    bestGenLoss = state.get("bestGenLoss", float("inf"))
    print(f"已恢复：epoch {state['epoch']}，bestGenLoss: {bestGenLoss:.4f}")
    return state["epoch"], state["history"], bestGenLoss


def train(train, test, gen, disc, discOpt, genOpt, startEpoch=1,
          history=None, bestGenLoss=None):
    bce = BCEWithLogitsLoss()
    l1 = L1Loss()

    discScaler = GradScaler(config.device)
    genScaler = GradScaler(config.device)

    if history is None:
        history = []
    if bestGenLoss is None:
        bestGenLoss = float("inf")

    gen.train()
    disc.train()
    currentEpoch = startEpoch - 1

    try:
        for epoch in range(startEpoch, config.epochs + 1):
            currentEpoch = epoch
            epochStart = time.time()
            params = {}
            params["genTotalLoss"] = 0
            params["discTotalLoss"] = 0

            looper = tqdm(train, leave=False, desc=f"[{epoch}/{config.epochs}]")

            for i, (l, ab) in enumerate(looper):
                gl, dl = trainStep(l, ab, gen, disc, discOpt, genOpt,
                                   genScaler, discScaler, bce, l1)
                params["genTotalLoss"] += gl / config.epochs
                params["discTotalLoss"] += dl / config.epochs
                looper.set_postfix(params)

            elapsed = round(time.time() - epochStart)
            print(f"[{epoch}/{config.epochs}] discLoss: {params['discTotalLoss']:.4f} "
                  f"genLoss: {params['genTotalLoss']:.4f} ({elapsed}s)")
            history.append(params)

            # 如果 genLoss 更低，保存最佳模型
            currentLoss = params["genTotalLoss"]
            if currentLoss < bestGenLoss:
                bestGenLoss = currentLoss
                torch.save(gen.state_dict(),
                           os.path.join(config.checkpointDirectory, "gen-best.pth"))
                torch.save(disc.state_dict(),
                           os.path.join(config.checkpointDirectory, "disc-best.pth"))
                print(f"  >> 新最佳模型 (genLoss: {bestGenLoss:.4f})")

            # 每个 epoch 保存断点
            save_checkpoint(gen, disc, genOpt, discOpt, genScaler, discScaler,
                            epoch, history, bestGenLoss, silent=True)

            # 定期导出带编号的备份
            if epoch % config.checkPointEvery == 0:
                torch.save(gen.state_dict(),
                           os.path.join(config.checkpointDirectory,
                                        config.genCheckPointTemplate.format(t=epoch)))
                torch.save(disc.state_dict(),
                           os.path.join(config.checkpointDirectory,
                                        config.discCheckPointTemplate.format(t=epoch)))

            # 每轮生成一张样例图
            with torch.no_grad():
                for x, y in test:
                    x = x.to(config.device)
                    images = gen.forward(x)
                    x += 1
                    x *= 50
                    images *= 128
                    rgb = lab_to_rgb(torch.cat((x, images), 1))
                    save_image(rgb.cpu(),
                               os.path.join(config.outputDirectory,
                                            f"generated_{epoch}.png"))
                    break

    except KeyboardInterrupt:
        print(f"\n\n>> Ctrl+C 中断！保存状态（epoch {currentEpoch}）...")
        save_checkpoint(gen, disc, genOpt, discOpt, genScaler, discScaler,
                        currentEpoch, history, bestGenLoss)
        print("训练已安全中断，下次运行将自动恢复。")

    # 加载最佳模型
    best_path = os.path.join(config.checkpointDirectory, "gen-best.pth")
    if os.path.exists(best_path):
        gen.load_state_dict(torch.load(best_path))

    return history, gen, disc


if __name__ == "__main__":
    generator = Generator(1, 2).to(config.device)
    discriminator = Discriminator(1, 2).to(config.device)

    discOptim = Adam(discriminator.parameters(), config.lr, config.betas)
    genOptim = Adam(generator.parameters(), config.lr, config.betas)

    trainDataLoader = getDataLoader(config.trainFolder, batchSize=config.batchSize,
                                    numWorks=config.numWorkers)
    testDataLoader = getDataLoader(config.testFolder, batchSize=config.batchSize,
                                   numWorks=config.numWorkers)

    genScaler = GradScaler(config.device)
    discScaler = GradScaler(config.device)

    if os.path.exists(config.resumeFile):
        print(f"发现训练存档: {config.resumeFile}")
        startEpoch, history, bestGenLoss = load_checkpoint(
            generator, discriminator, genOptim, discOptim, genScaler, discScaler)
        if startEpoch >= config.epochs:
            print(f"训练已完成（{startEpoch}/{config.epochs}），无需续训。")
            print(f"如需重新训练，请删除 {config.resumeFile} 后重试。")
            exit(0)
        startEpoch += 1
    else:
        startEpoch = 1
        history = []
        bestGenLoss = float("inf")
        print("未发现训练存档，从头开始训练。")

    history, trainedGen, trainedDisc = train(
        trainDataLoader, testDataLoader, generator, discriminator,
        discOptim, genOptim,
        startEpoch=startEpoch,
        history=history,
        bestGenLoss=bestGenLoss,
    )

    torch.save(trainedGen.state_dict(),
               os.path.join(config.checkpointDirectory,
                            config.genCheckPointTemplate.format(t="final")))
    torch.save(trainedDisc.state_dict(),
               os.path.join(config.checkpointDirectory,
                            config.discCheckPointTemplate.format(t="final")))

    with open("history.json", "w") as f:
        json.dump(history, f)

    if os.path.exists(config.resumeFile):
        print(f"\n训练结束。断点存档: {config.resumeFile}")
        print(f"如需重新训练，请删除该文件后重试。")
