import asyncio
import os

from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient
from agent_framework.openai import OpenAIChatClient
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


def build_agents() -> dict[str, Agent]:
    return {
        "John": Agent(
            client=OpenAIChatCompletionClient(
                model=os.getenv("GROQ_MODEL"),
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            ),
            name="John",
            instructions="너는 John이다. 토론 상대방의 의견에 동의하거나 반박하며 간결하게 한 문장으로 답하라.",
        ),
        "Rick": Agent(
            client=GeminiChatClient(
                model=os.getenv("GEMINI_MODEL"),
                api_key=os.getenv("GEMINI_API_KEY"),
            ),
            name="Rick",
            instructions="너는 Rick이다. 토론 상대방의 의견에 동의하거나 반박하며 간결하게 한 문장으로 답하라.",
        ),
        "Mike": Agent(
            client=OpenAIChatClient(
                model=os.getenv("OPENAI_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            name="Mike",
            instructions="너는 Mike이다. 토론 상대방의 의견에 동의하거나 반박하며 간결하게 한 문장으로 답하라.",
        ),
    }


async def main() -> None:
    agents = build_agents()

    # 사용자로부터 토론 주제 입력받기
    topic = input("토론할 주제를 입력하세요: ").strip()
    if not topic:
        topic = "인공지능이 예술을 할 수 있는가?"

    # 대화 기록 유지
    history = [f"토론 주제: {topic}"]

    try:
        round_count = 1
        while True:
            print(f"\n--- [라운드 {round_count}] ---")

            for name, agent in agents.items():
                # 지금까지의 대화 맥락을 프롬프트로 구성
                context_prompt = (
                    "지금까지의 대화 내용:\n" + "\n".join(history) +
                    f"\n\n위 대화를 참고하여 {name}로서 네 차례의 의견을 한 문장으로 말해줘."
                )

                # 서버 에러 및 Rate Limit 대응을 위한 재시도 로직
                max_retries = 3
                retry_delay = 2
                response_text = ""

                for attempt in range(max_retries):
                    try:
                        result = await agent.run(context_prompt)
                        response_text = result.text.strip()
                        break
                    except Exception as e:
                        print(f"\n⚠️ [{name}] 서버 에러 또는 제한 감지 (시도 {attempt+1}/{max_retries}): {e}")
                        if attempt == max_retries - 1:
                            print(f"❌ [{name}] 연결 실패로 이번 라운드 발언을 스킵합니다.")
                            response_text = "(시스템 과부하로 발언을 생성하지 못했습니다.)"
                        else:
                            # 잠시 대기 후 재시도 (점점 대기 시간을 늘림)
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2

                # 출력 및 히스토리 추가
                print(f"[{name}]: {response_text}")
                history.append(f"[{name}]: {response_text}")

            # 한 바퀴(라운드)가 끝난 후 사용자 개입 기회 제공
            user_intervention = await asyncio.to_thread(
                input,
                "\n[개입하기] 의견을 남기세요 (종료하려면 'end' 또는 'q' 입력, 엔터치면 토론 계속): "
            )

            clean_input = user_intervention.strip().lower()
            if clean_input in ["end", "q"]:
                print("\n[프로그램 종료] 사용자에 의해 토론이 정상적으로 중단되었습니다.")
                break

            if user_intervention.strip():
                print(f"[사용자 개입]: {user_intervention.strip()}")
                history.append(f"[사용자]: {user_intervention.strip()}")

            round_count += 1

    except KeyboardInterrupt:
        print("\n\n[프로그램 종료] 사용자에 의해 토론이 강제 중단되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())
