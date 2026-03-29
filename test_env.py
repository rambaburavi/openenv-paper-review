from environment import PaperReviewEnv, Action

env = PaperReviewEnv()

obs = env.reset()

print("Observation:", obs)

action = Action(
    domain="Computer Vision",
    keywords=["CNN", "segmentation"],
    decision="accept"
)

obs, reward, done, _ = env.step(action)

print("Reward:", reward)