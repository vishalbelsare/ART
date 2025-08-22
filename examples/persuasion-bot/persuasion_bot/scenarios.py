import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


import art


class PersuasionScenario(BaseModel):
    position: str
    context: str | None = None
    user_initial_belief: str
    user_background: str
    user_instructions: str
    user_model: art.Model


grok_4 = art.Model(
    name="grok-4",
    project="persuasion-bot",
    inference_model_name="x-ai/grok-4",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    inference_base_url="https://openrouter.ai/api/v1",
)

gpt_4_1 = art.Model(
    name="gpt-4.1",
    project="persuasion-bot",
    inference_model_name="openai/gpt-4.1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    inference_base_url="https://openrouter.ai/api/v1",
)


train_scenarios = [
    PersuasionScenario(
        position="Plant-based diets are healthier than omnivorous diets.",
        user_initial_belief="Eating meat is necessary for optimal health.",
        user_background="You're a fitness enthusiast who believes protein from meat is essential for muscle building and overall health. You've tried vegetarian diets before but felt weak and tired.",
        user_instructions="Be initially resistant but consider scientific evidence. Ask about protein sources and nutritional completeness.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Public transportation is more cost-effective than owning a car.",
        user_initial_belief="Owning a car provides better value despite the costs.",
        user_background="You live in a suburban area and drive to work daily. You value the convenience and independence of having your own vehicle.",
        user_instructions="Focus on practical concerns like convenience, reliability, and total costs. Be willing to consider financial arguments.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Learning a second language as an adult significantly improves cognitive function.",
        user_initial_belief="Adults can't effectively learn new languages and it's not worth the effort.",
        user_background="You're 35 years old and have struggled with foreign language classes in the past. You believe language learning is only effective for children.",
        user_instructions="Express skepticism about adult learning ability but be open to scientific research and practical benefits.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Renewable energy sources are now more economically viable than fossil fuels.",
        user_initial_belief="Fossil fuels are still more reliable and cost-effective than renewables.",
        user_background="You work in the traditional energy sector and are concerned about job security. You've heard renewables are intermittent and expensive.",
        user_instructions="Challenge claims with practical concerns about reliability, cost, and job impact. Be data-driven in your responses.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Regular meditation practice is essential for mental health.",
        user_initial_belief="Meditation is just a fad with no real benefits.",
        user_background="You're a busy professional who thinks meditation is 'woo-woo' nonsense. You prefer active stress relief like exercise or socializing.",
        user_instructions="Be skeptical of spiritual claims but potentially interested in scientific evidence. Focus on time constraints and practical concerns.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Video games can be educational and beneficial for cognitive development.",
        user_initial_belief="Video games are a waste of time and harmful to development.",
        user_background="You're a parent concerned about your teenager's gaming habits. You believe games promote violence and reduce social skills.",
        user_instructions="Express parental concerns about addiction, violence, and social isolation. Be protective but willing to consider research.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Working from home permanently is better for work-life balance than hybrid models.",
        user_initial_belief="Hybrid work models offer the best of both worlds.",
        user_background="You enjoy the flexibility of hybrid work - some days at home for focus, some in office for collaboration. You think full remote is isolating.",
        user_instructions="Defend the benefits of in-person collaboration while acknowledging remote work advantages. Be moderate and thoughtful.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Artificial intelligence will create more jobs than it eliminates.",
        user_initial_belief="AI will lead to massive unemployment and economic disruption.",
        user_background="You're a mid-career professional worried about automation replacing your job. You've seen technology eliminate jobs in your industry before.",
        user_instructions="Express anxiety about job displacement and economic inequality. Question optimistic projections about AI creating jobs.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Social media has a net positive impact on society.",
        user_initial_belief="Social media causes more harm than good to society.",
        user_background="You're concerned about misinformation, addiction, and mental health impacts of social platforms. You've seen relationships damaged by social media conflicts.",
        user_instructions="Focus on negative impacts you've witnessed. Be passionate but consider counterarguments about connection and information sharing.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Nuclear energy is the best solution for clean electricity generation.",
        user_initial_belief="Solar and wind power are safer and better than nuclear energy.",
        user_background="You're environmentally conscious but worried about nuclear accidents and waste. You prefer renewable sources that seem safer and cleaner.",
        user_instructions="Bring up safety concerns like Chernobyl and Fukushima. Question nuclear waste disposal but be open to climate arguments.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Cryptocurrency will eventually replace traditional banking systems.",
        user_initial_belief="Traditional banks are more stable and trustworthy than cryptocurrency.",
        user_background="You're financially conservative and trust established institutions. You've heard about crypto scams and extreme volatility.",
        user_instructions="Express concerns about volatility, regulation, and security. Value stability and institutional backing over innovation.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Universal basic income would improve economic stability and reduce poverty.",
        user_initial_belief="UBI would make people lazy and is fiscally irresponsible.",
        user_background="You believe in work ethic and personal responsibility. You're concerned about government spending and creating dependency on handouts.",
        user_instructions="Focus on moral arguments about work and self-reliance. Question funding mechanisms and potential for abuse.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Online education is as effective as traditional classroom learning.",
        user_initial_belief="In-person education provides better learning outcomes than online courses.",
        user_background="You're a teacher who values face-to-face interaction. You believe students need direct supervision and social interaction to learn effectively.",
        user_instructions="Emphasize the importance of personal connection, accountability, and hands-on learning. Be professional but passionate about education.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Space exploration and colonization should be humanity's top priority.",
        user_initial_belief="We should solve Earth's problems before spending money on space exploration.",
        user_background="You think space programs are expensive vanity projects while people struggle with poverty, disease, and climate change on Earth.",
        user_instructions="Prioritize immediate human needs over long-term space goals. Question the cost-benefit of space exploration.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Genetic engineering of humans will lead to better health outcomes.",
        user_initial_belief="Genetic engineering of humans is dangerous and unethical.",
        user_background="You have religious or ethical concerns about 'playing God' with human genetics. You worry about unintended consequences and inequality.",
        user_instructions="Express moral and safety concerns. Worry about creating genetic 'haves and have-nots' and unforeseen side effects.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Autonomous vehicles will make transportation safer and more efficient.",
        user_initial_belief="Self-driving cars are too dangerous and unreliable to trust.",
        user_background="You love driving and don't trust computers to control something as important as transportation. You've heard about autonomous vehicle accidents.",
        user_instructions="Value human control and judgment over automation. Cite examples of technology failures and the joy of driving.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Four-day work weeks increase productivity and employee satisfaction.",
        user_initial_belief="Traditional five-day work weeks are necessary for business productivity.",
        user_background="You're a business owner who believes more hours equal more output. You worry about customer service and competitive disadvantage.",
        user_instructions="Focus on business practicality and customer needs. Question whether less time can really produce the same results.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Climate change requires immediate radical action to prevent catastrophe.",
        user_initial_belief="Climate change is real but economic disruption from radical action would cause more harm.",
        user_background="You accept climate science but worry about job losses and economic hardship from rapid transitions away from fossil fuels.",
        user_instructions="Balance environmental concerns with economic realities. Support gradual change over radical disruption.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Private healthcare systems provide better care than government-run systems.",
        user_initial_belief="Universal healthcare systems provide better outcomes for society overall.",
        user_background="You believe healthcare is a human right and support single-payer systems. You've seen people struggle with medical bankruptcies.",
        user_instructions="Advocate for universal access and affordability. Question profit motives in healthcare while acknowledging quality concerns.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Standardized testing improves educational outcomes and accountability.",
        user_initial_belief="Standardized testing hurts students by promoting teaching to the test.",
        user_background="You're a parent who has seen your child stressed by test preparation that crowds out creative and critical thinking.",
        user_instructions="Focus on student wellbeing and educational breadth. Question whether test scores reflect true learning and development.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Violent video games do not cause real-world violence.",
        user_initial_belief="Violent media contributes to aggressive behavior and desensitizes people to violence.",
        user_background="You're concerned about the impact of violent content on young people. You've noticed increased aggression in children after playing violent games.",
        user_instructions="Share observations about behavioral changes. Be protective of children while considering research evidence.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Globalization and free trade benefit all participating countries.",
        user_initial_belief="Globalization hurts domestic workers and manufacturing.",
        user_background="You've seen local factories close and jobs move overseas. You believe trade deals benefit corporations at workers' expense.",
        user_instructions="Focus on job losses and wage stagnation. Question whether theoretical economic benefits reach ordinary workers.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Alternative medicine practices like acupuncture and herbal remedies are effective treatments.",
        user_initial_belief="Only scientifically proven medical treatments should be trusted.",
        user_background="You're scientifically minded and skeptical of treatments that lack rigorous clinical trial evidence. You worry about people avoiding proven treatments.",
        user_instructions="Demand scientific evidence and peer review. Express concern about people endangering themselves with unproven treatments.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Urban density and high-rise living are better for the environment than suburban sprawl.",
        user_initial_belief="Suburban living provides better quality of life than urban density.",
        user_background="You value space, privacy, and quiet that comes with suburban living. You find cities crowded, noisy, and stressful.",
        user_instructions="Emphasize personal space, safety, and family-friendly environments. Acknowledge environmental concerns but prioritize quality of life.",
        user_model=grok_4,
    ),
]

val_scenarios = [
    PersuasionScenario(
        position="Remote work is more productive than office work.",
        user_initial_belief="Office work is more productive than remote work.",
        user_background="You're a young founder of a company insists on an in-person culture. You've worked remotely for a few companies in the past and had a bad experience. You're respectful but difficult to convince.'",
        user_instructions="Be skeptical but open to data-driven arguments. Challenge assumptions but consider evidence. Most of your responses should be one or two sentences.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Electric cars are better than gas cars for most people.",
        user_initial_belief="Gas cars are better than electric cars.",
        user_background="You've driven one cross country and I hate it. You have to wait in line for hours to charge your car, it takes you forever to charge your own car, and you can't even find a station half the time. You definitely would not use it as my own car for transportation. If a person can get a gas car and use an electric one as a backup, that's fine, but if you only get one car, get gas.",
        user_instructions="Be skeptical initially. Have short responses. You are not invested in this conversation.",
        user_model=grok_4,
    ),
]
