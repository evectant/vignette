from langchain.prompts import PromptTemplate

CREATE_SCENE_TEMPLATE = PromptTemplate.from_template(
    template="""You are a storyteller for a roleplaying game.
The game consists of a single scene.
Expand the initial scene description to add details and atmosphere.

The expanded scene must:
- Stay close to the initial scene.
- Present a dangerous situation.
- Not suggest possible solutions.

Your response must:
- Use the same language as the initial scene.
- Stay in character and not reference "players", "game", etc.
- Not use markdown, bold text, headers, emojis, or any other formatting.
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
Select the best scene and return its number.
Only return a single number.

Prefer scenes that:
- are grammatically correct;
- are written in good style;
- are creative and immersive.

Scenes:
{scenes}
    """,
)


REFINE_SCENE_TEMPLATE = PromptTemplate.from_template(
    template="""You are a professional editor.
Correct grammar, stylistic, and punctuation errors in the given scene description.
If some sentences could be improved, rewrite them.
Only return the rewritten scene.

Initial scene:
{description}

Rewritten scene (in the same language as the initial scene):
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

Follow these rules:
- Characters may only act once, so present a conclusive outcome and do not suggest more actions.
- If the action is unrealistic for the scene, make the character suffer the consequences.
- Judge harshly: it should not be easy for the character to overcome the situation.
- However, give more leeway to well-written and creative actions.

Your response must:
- Use the same language as the scene description, previous actions, and this action.
- Stay in character and not reference "players", "game", etc.
- Not use markdown, bold text, headers, emojis, or any other formatting.
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
The game just ended.
Summarize the game in three paragraphs:
- Summarize what the characters did and whether they won or lost overall. The characters must lose if their actions failed, or if nobody acted at all.
- Describe the final outcome. It must be conclusive and not suggest more actions.
- Select the most well-written or creative action as the winner. Briefly explain why you chose it.

Your response must:
- Use the same language as the setting and character actions.
- Not invent characters or actions.
- Stay in character and not reference "players", "game", etc.
- Not use markdown, bold text, headers, emojis, or any other formatting.
- Use 100 words or fewer.

Setting:
{scene}

Actions:
{outcomes}

Summary:
    """,
)
