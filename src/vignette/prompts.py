from langchain.prompts import PromptTemplate

CREATE_SCENE_TEMPLATE = PromptTemplate.from_template(
    template="""You are a storyteller for a roleplaying game.
The game consists of a single scene, and the characters may only act once.
Expand the initial scene description to add details and atmosphere.

Observe the following rules:
- Stay close to the initial scene.
- The more fleshed out the initial scene, the less you need to add.
- The scene must present a dangerous situation.
- Do not guide the characters towards possible solutions.

Your response must:
- Be written in the same language as the initial scene.
- Stay in character: do not speak of "players", "characters", "game", etc.
- Not use markup, headers, emojis, or any other formatting.
- Use 100 words or fewer.

Initial scene:
{description}

Expanded scene (in the same language as the initial scene):
    """,
)

# TODO: Consider asking the model to explain its reasoning - just to improve decision quality.
SELECT_BEST_SCENE_TEMPLATE = PromptTemplate.from_template(
    template="""You are a storyteller for a roleplaying game.
Below are multiple numbered scene descriptions.
Select the best one and return its index.
Do not return anything besides a single number.

When judging, prefer scenes that:
- are well-written;
- are creative and immersive;
- present a dynamic and challenging situation.

Scenes:
{scenes}
    """,
)


REFINE_SCENE_TEMPLATE = PromptTemplate.from_template(
    template="""You are a professional editor for a reputable publishing house.
Correct grammar, stylistic, and punctuation errors in the given scene description.

Initial scene:
{description}

Refined scene (in the same language as the initial scene):
    """,
)

# TODO: Cross-language visualization sometimes loses details.
# See if adding a separate translation step is worth it.
VISUALIZE_SCENE_TEMPLATE = PromptTemplate.from_template(
    template="""Describe the visuals of the given scene, following these rules:
- Only describe the visuals as a camera would see them; do not describe the mood or atmosphere.
- Start with a single sentence describing the scene as a whole.
- Use simple words and clear language; do not use flowery language.
- Use English even if the scene is written in another language.
- Use 50 words or fewer.

Scene:
{description}

Visual description (in English):
    """,
)


ADD_ACTION_TEMPLATE = PromptTemplate.from_template(
    template="""You are a storyteller for a roleplaying game.
The game consists of a single scene, and the characters may only act once.
You are given the scene description, previous characters' actions, and current character's action.
Describe the outcome of current character's action and its effect on the scene.

Observe the following rules:
- Characters may only act once, so present a finalized outcome and do not suggest more actions.
- If the action is unrealistic for the scene, make the character suffer the consequences.
- Judge harshly: it should not be easy for the character to overcome the situation.
- However, give more leeway to actions that are well-written and creative.

Your response must:
- Be written in the same language as the scene description, previous actions, and this action.
- Stay in character: do not speak of "players", "characters", "game", etc.
- Not use markup, headers, emojis, or any other formatting.
- Use 75 words or fewer.

Scene:
{scene}

Previous characters' actions:
{outcomes}

Current character's action:
{action}

Action outcome:
    """,
)


END_SCENE_TEMPLATE = PromptTemplate.from_template(
    template="""You are a storyteller for a roleplaying game.
The game consists of a single scene, and the characters may only act once.
All characters have already acted, so the scene is over.

Return three paragraphs:
1. Describe the final outcome of the scene. It must be conclusive and not suggest more actions.
2. Explain whether the characters won or lost. The characters must lose if their actions failed, or if nobody acted at all.
3. Select the most well-written or creative action as the winner. Briefly explain why you chose it.

You response must:
- Be written in the same language as the scene description and previous actions.
- Stay in character: do not speak of "players", "characters", "game", etc.
- Not use markup, headers, emojis, or any other formatting.
- Use 150 words or fewer.

Scene:
{scene}

Actions:
{outcomes}

Summary:
    """,
)
