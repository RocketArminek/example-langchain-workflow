import datetime
import os
import json

from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain_community.document_loaders import RedditPostsLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

reddit = RedditPostsLoader(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent="researcher by s3b3q",
    categories=["new", "hot"],  # List of categories to load posts from
    mode="subreddit",
    search_queries=[
        "saas"
    ],  # List of subreddits to load posts from
    number_posts=50,  # Default value is 10
)

llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o")


def load_reddit(self, **kwargs):
    print(kwargs)
    return reddit.load()


def reddit_docs():
    content = []
    all = reddit.load()
    for doc in all:
        content.append(doc.page_content)
    return content


def workflow():
    filter_prompt = PromptTemplate(
        input_variables=["input"],
        template="You are researcher looking for ML related problems in SAAS industry."
                 "Given a list of posts from the subreddit, filter out the ones that are related to ML."
                 "Posts: {input}")
    filter_chain = LLMChain(llm=llm, prompt=filter_prompt, output_key="mlPosts", verbose=True)

    summarize_prompt = PromptTemplate(
        input_variables=["mlPosts"],
        template="You are a researcher working for SAAS."
                 "The company is aimed to build a revolutionary product in area of delivering ML models as a service."
                 "You are tasked with summarizing the newest posts from the given subreddit."
                 "Provide common problems other founder have. Came up with 3 main problems."
                 "Posts: {ml_posts}")
    summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt, output_key="problemList", verbose=True)

    cpo = PromptTemplate(
        input_variables=["problemList"],
        template="You are a CPO working for SAAS."
                 "For given problems write a set of questions that will help founders avoid such problems."
                 "Translate problems into feature descriptions."
                 "Plan their implementation in the product roadmap."
                 "Problems: {problemList}")
    cpo_chain = LLMChain(llm=llm, prompt=cpo, output_key="tasks", verbose=True)

    data_analyst = PromptTemplate(
        input_variables=["tasks"],
        template="You are a data analyst working for SAAS."
                 "To maximize efficiency of the product development process for each given task produce experiment that will help to validate the feature."
                 "Tasks: {tasks}")
    data_analyst_chain = LLMChain(llm=llm, prompt=data_analyst, output_key="experiments", verbose=True)

    md_to_html = PromptTemplate(
        input_variables=["experiments"],
        template="For given Markdown input generate HTML. Return pure HTML without any additional comments. "
                 "Input {experiments}")
    md_to_html_chain = LLMChain(llm=llm, prompt=md_to_html, output_key="html", verbose=True)

    general_chain = SimpleSequentialChain(
        chains=[filter_chain, summarize_chain, cpo_chain, data_analyst_chain, md_to_html_chain], verbose=True)
    result = general_chain.invoke({"input": reddit_docs()})

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    with open("result_" + timestamp + ".html", 'w') as f:
        f.write(result['output'])
        f.write("\n")


if __name__ == "__main__":
    workflow()