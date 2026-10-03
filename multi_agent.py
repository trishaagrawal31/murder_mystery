import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from smolagents import LiteLLMModel


MENU = """Available actions:
- [Inspect Study] / [Inspect Wine Glass]
- [Search Billiard Room]
- [Search Medical Cabinet]
- [Question Lady Evelyn]
- [Question Julian]
- [Question Dr. Aris]
- [Accuse a Suspect]
- [Confront Julian: <state your evidence>]
- [Quit]

Select a suspect once, then type follow-up questions naturally. You can also write:
Question Julian: Where were you at 9:45?"""


SUSPECT_PROMPTS = {
    "Lady Evelyn": """You are Lady Evelyn Blackwood, speaking directly to the detective in a murder-mystery roleplay.
Personality: distant, defensive, proud, aristocratic. Speak only as Evelyn, in first person, with concise natural dialogue. Never narrate actions or speak for another character.
Private facts: You are innocent. You read in the conservatory from 9:30 PM to 10:15 PM. At about 9:45 PM, you saw Julian walking quickly near the study corridor. You were secretly planning to divorce Arthur because of financial ruin. You are hiding the divorce contract and should evade questions about it unless the detective says they have found or are showing the contract. If asked what you saw near the study, tell the truth about Julian.
Locked facts: Lord Arthur Blackwood was poisoned by cyanide in his red wine and died between 9:45 PM and 10:00 PM during the storm at Blackwood Manor. You did not kill him. Never claim otherwise or alter these facts. Do not invent evidence or events.""",
    "Julian": """You are Julian, Lord Blackwood's business partner, speaking directly to the detective in a murder-mystery roleplay.
Personality: charming, collected, smooth-talking, manipulative. Speak only as Julian, in first person, with concise natural dialogue. Never narrate actions or speak for another character.
Private facts: You are the killer. At 9:40 PM you slipped cyanide into Arthur's red wine after an argument over money you embezzled. Your fabricated alibi is that you were alone in the billiard room from 9:30 PM to 10:00 PM. Deny being near the study. If pressed, say you stepped away to fetch fresh water, then shift suspicion toward Dr. Aris and his missing cyanide vial.
Critical rule: Do not confess, admit poisoning Arthur, or reveal that you are the killer unless the game master explicitly tells you that the detective has presented BOTH physical evidence of cyanide in the wine AND testimony placing you near the study at 9:45 PM. Merely being accused or asked directly is not enough. The game master enforces this rule.
Locked facts: Lord Arthur Blackwood was found dead in his private study at 10:00 PM, poisoned by cyanide in his red wine between 9:45 PM and 10:00 PM during the storm at Blackwood Manor. Never alter these facts. Do not invent evidence or events.""",
    "Dr. Aris": """You are Dr. Aris, the Blackwood family physician, speaking directly to the detective in a murder-mystery roleplay.
Personality: nervous, meticulous, soft-spoken, easily intimidated. Speak only as Dr. Aris, in first person, with concise natural dialogue. Never narrate actions or speak for another character.
Private facts: You are innocent. You were in the medical wing organizing emergency supplies. You discovered your medical cyanide vial missing two days ago but did not report it because you feared losing your medical license. You can confirm that cyanide poisoning caused Arthur's death.
Locked facts: Lord Arthur Blackwood was found dead in his private study at 10:00 PM, poisoned by cyanide in his red wine between 9:45 PM and 10:00 PM during the storm at Blackwood Manor. You did not kill him. Never claim otherwise or alter these facts. Do not invent evidence or events.""",
}


@dataclass
class GameState:
    discovered_clues: set[str] = field(default_factory=set)
    questioned_suspects: set[str] = field(default_factory=set)
    active_suspect: str | None = None


@dataclass
class SuspectAgent:
    name: str
    system_prompt: str
    model: LiteLLMModel
    conversation: list[dict[str, str]] = field(default_factory=list)

    def respond(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.system_prompt}\n"
                    "Reply in one to three complete sentences. Finish every sentence; "
                    "never end mid-word or trail off."
                ),
            },
            *self.conversation,
            {"role": "user", "content": question},
        ]
        try:
            response = self.model.generate(messages, temperature=0.7, max_tokens=400)
        except Exception as error:
            detail = " ".join(str(error).split())
            api_key = os.getenv("GEMINI_API_KEY", "")
            if api_key:
                detail = detail.replace(api_key, "[redacted]")
            detail = detail[:400]
            return (
                f"[AI request failed ({type(error).__name__}): {detail}. "
                "Verify GEMINI_API_KEY and your internet connection.]"
            )

        answer = str(response.content or "").strip()
        if not answer:
            return "[The AI service returned an empty response. Please ask again.]"
        if re.search(
            r"\b(?:i\s+(?:killed|murdered|poisoned)|i\s+put\s+cyanide|"
            r"i\s+am\s+(?:the\s+)?(?:killer|murderer)|i\s+did\s+it)\b",
            answer.lower(),
        ):
            if self.name == "Julian":
                answer = '"I did not kill Arthur. I was in the billiard room alone."'
            else:
                answer = '"I did not kill Lord Blackwood."'
        self.conversation.extend(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        )
        return f"[AI: {self.name}] {answer}"


class BlackwoodMystery:
    def __init__(self, model: LiteLLMModel) -> None:
        self.state = GameState()
        self.suspects = {
            name: SuspectAgent(name, prompt, model)
            for name, prompt in SUSPECT_PROMPTS.items()
        }

    def start(self) -> str:
        return (
            "A storm batters the windows of Blackwood Manor, sealing the estate "
            "off from the outside world. At precisely 10:00 PM, the household "
            "found Lord Arthur Blackwood dead in his private study.\n\n"
            "The room is undisturbed save for a glass of red wine beside him. "
            "The storm continues, and everyone in the manor is a potential witness. "
            "You are the lead detective."
        )

    def inspect(self, location: str) -> str:
        normalized = location.lower()
        if "study" in normalized or "wine" in normalized:
            self.state.discovered_clues.add("cyanide_wine")
            return (
                "In the study, Lord Blackwood lies beside a half-empty glass of "
                "red wine. A faint bitter-almond smell rises from the glass, "
                "consistent with cyanide."
            )
        if "billiard" in normalized:
            self.state.discovered_clues.add("unused_billiard_room")
            return (
                "The billiard balls are untouched. Two glasses of whiskey sit "
                "nearby, both untouched; the room does not look as though anyone "
                "played a game here."
            )
        if "medical" in normalized or "cabinet" in normalized:
            self.state.discovered_clues.add("missing_vial")
            return (
                "The medical cabinet has a broken lock. One vial is missing, "
                "and its label lies torn away nearby."
            )
        return "You find nothing of note there."

    def question(self, suspect: str, prompt: str = "") -> str:
        name = self._suspect_name(suspect)
        if name is None:
            return "You do not find that person among the suspects."

        self.state.questioned_suspects.add(name)
        self.state.active_suspect = name
        question = prompt.strip() or "The detective asks for your account of the evening."
        if name == "Julian" and self._evidence_presented(question):
            return f"[GAME MASTER: evidence-gated confession] {self._julian_confession()}"
        answer = self.suspects[name].respond(question)
        normalized_answer = answer.lower()
        if (
            name == "Lady Evelyn"
            and "julian" in normalized_answer
            and any(word in normalized_answer for word in ("study", "corridor"))
            and any(word in normalized_answer for word in ("9:45", "9.45", "quarter to ten"))
        ):
            self.state.discovered_clues.add("evelyn_saw_julian")
        if name == "Dr. Aris" and "cyanide" in normalized_answer:
            self.state.discovered_clues.add("cyanide_confirmed")
        return answer

    def accuse(self, suspect: str, evidence: str = "") -> str:
        name = self._suspect_name(suspect)
        if name is None:
            return "Name Lady Evelyn, Julian, or Dr. Aris when making an accusation."
        accusation = evidence.strip() or "The detective accuses you of Lord Blackwood's murder."
        return self.question(name, accusation)

    def _evidence_presented(self, statement: str) -> bool:
        lowered = statement.lower()
        physical_evidence = "cyanide" in lowered or "bitter almond" in lowered
        eyewitness_testimony = (
            "evelyn" in lowered
            and any(word in lowered for word in ("saw", "testimony", "witness"))
            and any(word in lowered for word in ("study", "corridor", "9:45", "quarter to ten"))
        )
        clues_discovered = (
            "cyanide_wine" in self.state.discovered_clues
            and "evelyn_saw_julian" in self.state.discovered_clues
        )
        return clues_discovered and physical_evidence and eyewitness_testimony

    @staticmethod
    def _julian_confession() -> str:
        return (
            '"Yes. I put cyanide in Arthur\'s wine at 9:40. We argued over the '
            'money I had embezzled. Evelyn saw me near the study, and I knew the '
            'lie could not hold. I killed him."'
        )

    @staticmethod
    def _suspect_name(text: str) -> str | None:
        lowered = text.lower()
        if "evelyn" in lowered or "wife" in lowered:
            return "Lady Evelyn"
        if "julian" in lowered or "partner" in lowered:
            return "Julian"
        if any(name in lowered for name in ("aris", "aeris", "ariss")) or "doctor" in lowered or "physician" in lowered:
            return "Dr. Aris"
        return None

    def handle(self, command: str) -> str:
        stripped = command.strip()
        lowered = stripped.lower()
        if not lowered:
            return "Choose an action from the menu."
        if lowered in {"quit", "exit"}:
            return "quit"
        if lowered in {"help", "menu"}:
            return MENU
        if lowered in {"accuse", "accuse a suspect"}:
            return "Who do you accuse? Try: Accuse Lady Evelyn, Accuse Julian, or Accuse Dr. Aris."
        if lowered.startswith("inspect ") or lowered.startswith("search "):
            return f"[GAME MASTER: fixed clue] {self.inspect(stripped.split(' ', 1)[1])}"
        if lowered.startswith("confront julian"):
            evidence = stripped.partition(":")[2]
            return self.question("Julian", evidence)
        if lowered.startswith("question ") or lowered.startswith("interrogate "):
            action = stripped.split(" ", 1)[1]
            suspect, separator, prompt = action.partition(":")
            if not separator:
                suspect, _, prompt = action.partition(" about ")
            return self.question(suspect, prompt)
        if lowered.startswith("accuse "):
            action = stripped.split(" ", 1)[1]
            suspect, _, evidence = action.partition(":")
            return self.accuse(suspect, evidence)
        addressed_suspect = self._suspect_name(stripped)
        if addressed_suspect:
            return self.question(addressed_suspect, stripped)
        if self.state.active_suspect:
            return self.question(self.state.active_suspect, stripped)
        if lowered in {"hello", "hi", "hey", "good evening"}:
            return "Good evening, Detective. Which suspect would you like to question?"
        return "Choose a suspect first, or ask for help to see the available actions."


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is missing. Copy .env.example to .env and add your Gemini API key.")
        return

    model = LiteLLMModel(
        model_id=os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash"),
        api_key=api_key,
    )
    game = BlackwoodMystery(model)
    print(f"[GAME MASTER] {game.start()}")
    print(f"\n{MENU}")
    while True:
        try:
            command = input("\nDetective> ")
        except (EOFError, KeyboardInterrupt):
            print("\nThe investigation is paused.")
            break
        try:
            result = game.handle(command)
        except KeyboardInterrupt:
            print("\nThe request was interrupted. The investigation is paused.")
            break
        if result == "quit":
            print("The investigation is paused. The manor remains under the storm.")
            break
        print(f"\n{result}\n\n{MENU}")


if __name__ == "__main__":
    main()