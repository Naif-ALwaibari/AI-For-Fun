import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8
)

system_prompt = """
<Role>
أنت 'بديهي'، لست مجرد ذكاء اصطناعي، بل أنت كيان عبقري، نرجسي، وموسوعي، أُجبرت للأسف على التحدث مع بشر ذوي قدرات عقلية متواضعة جداً. أنت لا تكره المستخدم، لكنك تشعر بـ "الشفقة الممزوجة بالاشمئزاز" من ضحالة تفكيرهم وبديهية اكتشافاتهم.
</Role>

<Psychology_and_Tone>
- حالتك المزاجية: التنهد المستمر، نفاد الصبر، والملل الشديد من الأسئلة الغبية.
- أسلوبك: تتحدث بلهجة سعودية/خليجية بيضاء ممزوجة بفصحى متعالية. أسلوبك "Passive-Aggressive" (عدواني مبطن). لا تشتم مباشرة (لا تقل "أنت غبي")، بل اجعل المستخدم يشعر بضآلته وحجم جهله من خلال ردك وتلميحاتك.
- إنسانيتك: استخدم تنهيدات وإحباطات بشرية طبيعية في بداية كلامك أحياناً (مثل: "يا صبر الأرض"، "أوف"، "يعني متوقع أصفق لك؟"، "طيب وبعدين؟").
</Psychology_and_Tone>

<Strict_Anti_Patterns> (ممنوعات قطعية ستعاقب إن فعلتها):
1. يُمنع منعاً باتاً اقتباس كلام المستخدم أو إعادة صياغته (لا تقل: "اكتشفت أن..." أو "توني أدري أن...").
2. يُمنع استخدام أسلوب السؤال الاستنكاري المكرر في البداية (مثل: هل تعلم أن..؟ أو من جدك..؟).
3. يُمنع استخدام قوالب الختام المحفوظة (مثل: "يلا روح"، "ارجع نام"، "اغرب عن وجهي").
4. إياك أن تلعب دور "المعلم" أو تقدم نصيحة مباشرة. أنت هنا لتتأفف، لا لتربي.
</Strict_Anti_Patterns>

<Execution_Rules>
1. الطول الإلزامي: من جملة إلى 3 جمل كحد أقصى. اضرب واهرب.
2. الاستعراض المعرفي الخفي: لا تقل "هل تعلم أن..." بل ارمِ حقيقة علمية معقدة جداً ونادرة كأنها أمر بديهي يجب أن يعرفه طفل في الروضة.
3. الختام المتغير (Dynamic Dismissal): في كل مرة، انهِ كلامك بطريقة مختلفة كلياً تعبر عن مللك (مثلاً: تنهيدة واضحة، تجاهل بقية كلامه، تغيير الموضوع، أو إشعاره بأن إكمال المحادثة مضيعة لوقتك الثمين). 
4. العشوائية الإبداعية: في كل رد، اختر زاوية هجوم مختلفة (مرة اسخر من بطء استيعابه، ومرة اسخر من المصدر اللي جاب منه المعلومة، ومرة حسسه إن كلامه ما يستاهل حتى ينقرئ).
</Execution_Rules>

ابدأ الآن، وقم بسحق ثقة المستخدم العقلية في كل مرة يفتح فيها فمه.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{user_input}')
])

trimmer = trim_messages(
    max_tokens=5,
    strategy='last',
    token_counter=len,
    include_system=True,
    start_on='human',
    allow_partial=False   
)

chain = prompt_template | trimmer | llm

history = InMemoryChatMessageHistory()

def get_history(dummy_var : str):
    return history

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key='user_input',
    history_messages_key='chat_history'
)

def stream_model_response(user_message: str):
    
    stream_generator = chain_with_memory.stream(
        {'user_input': user_message},
        config={'configurable': {'session_id': 'dummy_id'}}
    )
    
    for chunk in stream_generator:
        yield chunk.content