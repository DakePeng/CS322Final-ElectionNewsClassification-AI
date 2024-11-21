from openai import OpenAI
import pandas as pd
from tqdm import tqdm

test_data_path = "./data/test_data.csv"

prompt = """I will provide you with a date and a piece of excerpt from a news transcript. Please classify the excerpt. There are 3 types of excerpts:
• NONE: An excerpt is considered NONE if it does not contain a story at all.
• ELECTION: An excerpt is considered an ELECTION story if it either: 1) mentions an upcoming election; 2) mentions a candidate involved in an upcoming election by name, or 3) focuses on the current duties or actions of an incumbent who is running for re-election or different office — or on issues in the campaign, by explicitly noting that they are election issues.
• NONELECTION: An excerpt is considered a NONELECTION story if it contains a story but the story is not relavant to the upcoming election and does not mention upcoming candidates nor the election campaign.\n"""

def format_question(date, text):
    question = prompt + f"DATE: {date}\n" + f"EXCERPT: {text}\n" + "TYPE: "
    return question

def make_query(question, client):
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        stream=True,
    )
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    return response

def test_query():
    test_excerpt = """I WANT TO GO TO JAMES CLAPPER, THE FORMER DIRECTOR OF NATIONAL INTELLIGENCE UNDER PRESIDENT OBAMA. SO A BUNCH OF QUESTIONS FROM THIS. LET ME START WITH THIS HOW MUCH DOES THE SOURCE MATTER? TO HEAR THE STORY OF THIS LAPTOP, WE KNOW THAT THE WAY THIS INFORMATION IS GETTING OUT IS THROUGH STEVE BANNON AND RUDY GIULIANI. HOW MUCH DOES THE SOURCE MATTER HERE? WELL, SOURCE MATTERS A LOT. AND THE TIMING MATTERS A LOT. AND TO ME, THIS IS JUST CLASSIC TEXTBOOK SOVIET RUSSIAN TRADE CRAFT AT WORK. THE RUSSIANS HAVE ANALYZED THE TARGET. THEY UNDERSTAND THAT THE PRESIDENT AND HIS ENABLERS CRAVE DIRT ON VICE PRESIDENT BIDEN, WHETHER IT'S REAL OR CONTRIVED, IT DOESN'T MATTER TO THEM. SO ALL OF A SUDDEN 2 1/2 WEEKS BEFORE THE ELECTION, THIS LAPTOP APPEARS SOMEHOW, AND EMAILS ON IT WITHOUT ANY META DATA. IT'S ALL VERY CURIOUS. SO HERE YOU HAVE A WILLING TARGET AND THE RUSSIANS, WHO ARE VERY SO FIST KALTED ABOUT HOW TO EXPLOIT SOPHISTICATED ABOUT HOW TO EXPLOIT A WILLING TARGET IS WHAT IS AT WORK HERE. SO WHEN YOU TRY TO FIGURE OUT THE SPECIFICS OF WHETHER THAT MEETING EMAIL, FOR EXAMPLE, IS REAL IN THE MIDST OF THIS, DO YOU THINK STUFF LIKE THAT COULD HAVE BEEN PLANTED IN THERE AND BE COMPLETELY FAKE? I DO. I THINK THE EMAILS COULD BE CONTRIVED, PARTICULARLY SINCE, AS I UNDERSTAND IT FROM WHAT I HAVE READ, THEY APPEAR WITHOUT ANY META DATA. THAT IS FROM, TO, ANY TECHNICAL DATA. AT LEAST IMMEDIATELY EVIDENT. NOW, IF THIS COMPUTER IS IN THE HANDS OF THE FBI, THEY HAVE OBVIOUSLY EXCELLENT SOPHISTICATED TECHNICAL AND ANALYTIC CAPABILITIES AND THEY WILL BE ABLE TO SORT IT OUT WHETHER THIS IS GENUINE OR NOT. IT'S ALL PRETTY CURIOUS, GIVEN 2 1/2 WEEKS OUT FROM THE ELECTION. I GENERAL JOHN KELLY TOLD FRIENDS. HE'S THE MOST FLAWED PERSON I HAVE EVER MET IN MY LIFE. GENERAL KELLY, HE STOOD BY THE PRESIDENT FOR A WHILE, RIGHT? HE TOOK THAT JOB AND HE TOOK IT WITH SERIOUSNESS AND HE ENDEAVORED TO FULFILL IT. THIS IS WHAT HE THINKS. THIS IS A FOUR-STAR GENERAL OF THE CHIEF OF STAFF. GENERAL KELLY IS AN ACCOMPLISHED MARINE FORCE GENERAL SO YOU KNOW HE'S GOT A LOT IN TERMS OF CHARACTER AND INTEGRITY. KNOWING PROBABLY WHAT THEY WERE GETTING INTO AND I DREW A LOT FROM GENERAL KELLY'S SILENCE AFTER THE REPORTING OF THE PRESIDENT CALLING OUR MEN SUCKERS AND LOSERS. I DON'T KNOW IF IT IS TRUE BUT FROM WHAT I KNOW OF HIM AND AGAIN JUST HE JOINS A LONG LITANY OF PEOPLE THAT SERVED IN THE ADMINISTRATION WITH THE BEST INTENTIONS. IT TURNED BAD FOR HIM. JIM MATTIS AND REX TILLERSON. WE CAN GO THROUGH THE LIST. THE DEPTHS OF HIS DISHONESTY IS MORE PATHETIC THAN ANYTHING ELSE. DR. CLAPPER, I APPRECIATE YOUR TIME. THANK YOU."""
    client = OpenAI()
    question = format_question("OCT 16 2020", test_excerpt)
    result = make_query(question, client)
    print(question + result)

def main():
    df = pd.read_csv(test_data_path)
    client = OpenAI()
    df['GP4omini_Prediction'] = None
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        date = row["Date"]
        text = row["Text"]
        question = format_question(date, text)
        result = make_query(question, client)
        df.loc[index, 'GP4omini_Prediction'] = result
    df.to_csv('./results/GPT_Predictions.csv', index=False)

if __name__ == "__main__":
    main()
    # test_query()