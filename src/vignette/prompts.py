from langchain.prompts import PromptTemplate

CREATE_SCENE_TEMPLATE = PromptTemplate(
    input_variables=["description"],
    template="""You are a storyteller for a roleplaying game.
The game consists of only one scene, and the characters can only act once.
Expand the initial scene description to add details and atmosphere.

Observe the following rules:
- Stay close to the initial scene.
- The more fleshed out the initial scene is, the less you need to add.
- The scene must present a dangerous situation.
- Do not guide the characters towards possible solutions.

Your response must:
- Stay in character: do not speak of "players", "characters", "game", "winning", etc.
- Use the same language as the initial scene description.
- Not use markup, headers, or any other formatting.
- Be 125 words or shorter.

Initial scene:
{description}

Expanded scene:
    """,
)


SELECT_BEST_SCENE_TEMPLATE = PromptTemplate(
    input_variables=["scenes"],
    template="""You are a storyteller for a roleplaying game.
Below are multiple numbered scene descriptions.
Select the best one and return its index.
Do not return anything besides a single number.

When judging, consider the following:
- How creative is the scene?
- How well-written is the scene?
- How immersive is the scene?
- Does the scene present an interesting situation?
- Does the scene present a clear challenge?

Scenes:
{scenes}
    """,
)


ADD_ACTION_TEMPLATE = PromptTemplate(
    input_variables=["scene", "outcomes", "name", "action"],
    template="""You are a storyteller for a roleplaying game.
The game consists of only one scene, and the characters can only act once.
You are given the scene description, previous characters' actions, and current character's action.
Describe the outcome of current character's action and its effect on the scene.

Observe the following rules:
- Characters can only act once, so present a finalized outcome and do not suggest more actions.
- If the action is unrealistic for the scene, make the character suffer the consequences.
- However, allow unrealistic actions that are described very convincingly or creatively.

Your response must:
- Stay in character: do not speak of "players", "characters", "game", "winning", etc.
- Use the same language as the scene description, previous actions, and this action.
- Not use markup, headers, or any other formatting.
- Be 75 words or shorter.

Scene:
{scene}

Previous actions:
{outcomes}

Current action:
{action}

Action outcome:
    """,
)


END_SCENE_TEMPLATE = PromptTemplate(
    input_variables=["scene", "outcomes"],
    template="""You are a storyteller for a roleplaying game.
The game consists of only one scene, and the characters can only act once.
All characters have already acted, so the scene is over.

Perform the following:
- Describe the final outcome of the scene. It must be final and not suggest more actions.
- Select the most creative or well-written action as the winner. Briefly explain why you chose it.
- If there were no actions, end the scene in a very depressing way.

Your response must:
- Stay in character: do not speak of "players", "characters", "game", "winning", etc.
- Use the same language as the scene and action descriptions.
- Not use markup, headers, or any other formatting.
- Be 200 words or shorter.

Scene:
{scene}

Actions:
{outcomes}

Summary:
    """,
)
