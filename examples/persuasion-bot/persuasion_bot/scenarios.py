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

gemini_2_5_flash = art.Model(
    name="gemini-2.5-flash",
    project="persuasion-bot",
    inference_model_name="google/gemini-2.5-flash",
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
        position="Omnivorous diets are healthier than plant-based diets.",
        user_initial_belief="Plant-based diets provide optimal nutrition and health.",
        user_background="You've been vegetarian for years and feel great. You believe meat consumption leads to health problems and environmental damage.",
        user_instructions="Defend plant-based nutrition but be open to evidence about potential deficiencies or benefits of including animal products.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Public transportation is more cost-effective than owning a car.",
        user_initial_belief="Owning a car provides better value despite the costs.",
        user_background="You live in a suburban area and drive to work daily. You value the convenience and independence of having your own vehicle.",
        user_instructions="Focus on practical concerns like convenience, reliability, and total costs. Be willing to consider financial arguments.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Owning a car is more cost-effective than using public transportation.",
        user_initial_belief="Public transportation saves money compared to car ownership.",
        user_background="You live in a city with good transit and don't own a car. You use buses and trains daily and think cars are expensive and unnecessary.",
        user_instructions="Emphasize savings from no car payments, insurance, or maintenance. Be skeptical of hidden costs in car ownership calculations.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Learning a second language as an adult significantly improves cognitive function.",
        user_initial_belief="Adults can't effectively learn new languages and it's not worth the effort.",
        user_background="You're 35 years old and have struggled with foreign language classes in the past. You believe language learning is only effective for children.",
        user_instructions="Express skepticism about adult learning ability but be open to scientific research and practical benefits.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Adult language learning is ineffective and cognitive benefits are overstated.",
        user_initial_belief="Learning languages as an adult provides significant cognitive and career benefits.",
        user_background="You're a language teacher who has successfully learned multiple languages as an adult. You believe it's never too late to learn.",
        user_instructions="Share success stories and research on neuroplasticity. Question claims about age limitations in language acquisition.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Renewable energy sources are now more economically viable than fossil fuels.",
        user_initial_belief="Fossil fuels are still more reliable and cost-effective than renewables.",
        user_background="You work in the traditional energy sector and are concerned about job security. You've heard renewables are intermittent and expensive.",
        user_instructions="Challenge claims with practical concerns about reliability, cost, and job impact. Be data-driven in your responses.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Fossil fuels are more economically reliable than renewable energy sources.",
        user_initial_belief="Renewable energy is now cheaper and more sustainable than fossil fuels.",
        user_background="You're an environmental advocate who supports clean energy. You believe renewables are the future and fossil fuels are outdated.",
        user_instructions="Emphasize environmental benefits and falling renewable costs. Challenge claims about reliability and intermittency.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Regular meditation practice is essential for mental health.",
        user_initial_belief="Meditation is just a fad with no real benefits.",
        user_background="You're a busy professional who thinks meditation is 'woo-woo' nonsense. You prefer active stress relief like exercise or socializing.",
        user_instructions="Be skeptical of spiritual claims but potentially interested in scientific evidence. Focus on time constraints and practical concerns.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Meditation is overhyped and exercise is more effective for mental health.",
        user_initial_belief="Meditation is a proven, essential practice for mental wellness.",
        user_background="You've practiced meditation for years and credit it with reducing your anxiety and improving focus. You recommend it to everyone.",
        user_instructions="Share personal benefits and defend meditation research. Question whether physical exercise alone can provide the same mental benefits.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Video games can be educational and beneficial for cognitive development.",
        user_initial_belief="Video games are a waste of time and harmful to development.",
        user_background="You're a parent concerned about your teenager's gaming habits. You believe games promote violence and reduce social skills.",
        user_instructions="Express parental concerns about addiction, violence, and social isolation. Be protective but willing to consider research.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Video games are harmful to development and promote antisocial behavior.",
        user_initial_belief="Video games can be educational and beneficial for learning.",
        user_background="You're an educator who uses games in teaching and has seen students develop problem-solving skills through gaming.",
        user_instructions="Defend educational gaming and cognitive benefits. Question overly broad claims about violence and social harm.",
        user_model=gpt_4_1,
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
    PersuasionScenario(
        position="Suburban sprawl provides better quality of life than urban density.",
        user_initial_belief="Dense urban living is more sustainable and environmentally friendly.",
        user_background="You live in a compact city apartment and appreciate walkability, public transit, and cultural amenities. You think suburbs are wasteful.",
        user_instructions="Defend urban sustainability and efficiency. Question suburban environmental impact and social isolation.",
        user_model=gpt_4_1,
    ),
    # Add all the remaining missing opposites
    PersuasionScenario(
        position="Hybrid work models are better for work-life balance than permanent remote work.",
        user_initial_belief="Working from home permanently provides the best work-life balance.",
        user_background="You love working remotely and never want to go back to an office. You're more productive at home and appreciate the flexibility.",
        user_instructions="Defend remote work benefits like no commute and flexibility. Question the necessity of in-person collaboration.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Artificial intelligence will eliminate more jobs than it creates.",
        user_initial_belief="AI will create new opportunities and enhance human productivity.",
        user_background="You work in tech and see AI as a powerful tool that augments human capabilities rather than replacing them.",
        user_instructions="Emphasize AI as a collaborative tool and point to historical technology adoption creating new industries.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Social media causes more harm than good to society.",
        user_initial_belief="Social media connects people and democratizes information sharing.",
        user_background="You use social platforms to stay connected with friends and family. You've built meaningful online communities and relationships.",
        user_instructions="Highlight connection benefits and information access. Acknowledge problems but emphasize positive impacts.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Solar and wind power are better solutions than nuclear energy for clean electricity.",
        user_initial_belief="Nuclear energy is the most reliable clean power source.",
        user_background="You support nuclear power as a proven, scalable clean energy solution. You think renewables are intermittent and unrealistic.",
        user_instructions="Defend nuclear safety record and reliability. Question renewable energy storage and grid stability challenges.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Traditional banking systems are more stable than cryptocurrency.",
        user_initial_belief="Cryptocurrency represents the future of decentralized finance.",
        user_background="You're enthusiastic about blockchain technology and own various cryptocurrencies. You believe in financial decentralization.",
        user_instructions="Advocate for decentralization and innovation. Question traditional banking fees and government control.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Universal basic income would create dependency and reduce work motivation.",
        user_initial_belief="UBI would provide security and enable people to pursue meaningful work.",
        user_background="You support UBI as a solution to technological unemployment and economic inequality. You believe it would reduce poverty.",
        user_instructions="Emphasize freedom from economic insecurity and potential for innovation. Question assumptions about work motivation.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Traditional classroom learning is more effective than online education.",
        user_initial_belief="Online education provides flexibility and accessibility that traditional schools can't match.",
        user_background="You've completed online courses and degrees successfully. You appreciate self-paced learning and global access to education.",
        user_instructions="Highlight accessibility and flexibility benefits. Question whether in-person interaction is always necessary for learning.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Earth's problems should be prioritized over space exploration.",
        user_initial_belief="Space exploration drives innovation and ensures humanity's long-term survival.",
        user_background="You're fascinated by space technology and believe space exploration leads to beneficial innovations for Earth.",
        user_instructions="Emphasize technological spinoffs and long-term survival. Question short-term thinking about resource allocation.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Genetic engineering of humans is dangerous and ethically problematic.",
        user_initial_belief="Genetic engineering can eliminate hereditary diseases and improve human health.",
        user_background="You have a genetic condition that could be prevented in future generations through genetic engineering.",
        user_instructions="Emphasize potential to eliminate suffering and improve quality of life. Question ethical objections when weighed against benefits.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Human drivers are safer and more reliable than autonomous vehicles.",
        user_initial_belief="Self-driving cars will reduce accidents caused by human error.",
        user_background="You're excited about autonomous vehicle technology and believe computers can react faster and more consistently than humans.",
        user_instructions="Point to human error statistics and computer precision. Question emotional attachments to manual control.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Traditional five-day work weeks are more productive than four-day weeks.",
        user_initial_belief="Four-day work weeks improve productivity and employee wellbeing.",
        user_background="You've experienced or heard about successful four-day work week trials and believe shorter weeks reduce burnout.",
        user_instructions="Share examples of successful implementations and productivity research. Challenge traditional assumptions about time and output.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Gradual climate action is more practical than radical immediate changes.",
        user_initial_belief="Climate change is an emergency requiring immediate dramatic action.",
        user_background="You're deeply concerned about climate change and believe incremental changes are insufficient given the urgency.",
        user_instructions="Emphasize scientific urgency and tipping points. Question whether gradual change can address the scale of the problem.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Government-run healthcare systems provide better outcomes than private systems.",
        user_initial_belief="Private healthcare provides higher quality care and more choices.",
        user_background="You have good private insurance and have received excellent care. You worry about government healthcare being bureaucratic and slow.",
        user_instructions="Value choice and quality of care. Express concerns about government efficiency and medical innovation.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Standardized testing reduces educational quality and student wellbeing.",
        user_initial_belief="Standardized tests provide important accountability and measure student progress.",
        user_background="You're an education administrator who believes testing helps identify struggling schools and students who need support.",
        user_instructions="Defend the need for objective measurement and accountability. Question how else to ensure educational equity.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Violent video games contribute to aggressive behavior and desensitization.",
        user_initial_belief="Video game violence doesn't cause real-world aggression.",
        user_background="You're an avid gamer who plays violent games but considers yourself peaceful. You think games are clearly fantasy.",
        user_instructions="Distinguish between fantasy violence and real aggression. Question causal connections and point to research.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Globalization hurts domestic workers and increases inequality.",
        user_initial_belief="Free trade and globalization create prosperity and lift people out of poverty worldwide.",
        user_background="You study international economics and believe trade creates mutual benefits and economic growth globally.",
        user_instructions="Emphasize global poverty reduction and economic growth. Question protectionist policies and their effects.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Only scientifically proven medical treatments should be trusted.",
        user_initial_belief="Alternative medicine practices can be effective alongside conventional treatments.",
        user_background="You've experienced benefits from acupuncture or herbal remedies when conventional medicine didn't help your condition.",
        user_instructions="Share personal experiences and question why Western medicine dismisses traditional practices. Advocate for integrative approaches.",
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
    PersuasionScenario(
        position="Developers should star the OpenPipe ART repository.",
        user_initial_belief="Ambivalent.",
        user_background="You're a developer who has no idea what ART is or why it's important",
        user_instructions="Be skeptical about the pluggy nature of this argument. You're not invested in this conversation. Answer in low-effort, short responses.",
        context="""ART (Agent Reinforcement Trainer) is a library for building AI agents put out earlier in 2025 to help developers apply RL to their agents. The library has several thousand stars on GitHub. It's a great way to get started with RL.
        """,
        user_model=grok_4,
    ),
]
