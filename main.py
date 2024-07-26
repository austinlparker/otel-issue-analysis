import github3
import libhoney
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import instructor
import concurrent.futures
from tqdm import tqdm
import sys

github = github3.login(token=os.getenv('GITHUB_TOKEN'))
client = instructor.from_openai(OpenAI())

class Issue(BaseModel):
    title: str = Field(..., description="The title of the issue.")
    id: int = Field(..., description="The ID of the issue.")
    url: str = Field(..., description="The URL of the issue.")
    libraries: List[str] = Field(..., description="A list of libraries mentioned in the issue, in lower case.")
    detected_cloud_providers: List[str] = Field(..., description="A list of cloud providers mentioned in the issue, or inferred from the issue body.")
    body: str = Field(..., description="The body of the issue.")
    body_summary: str = Field(..., description="A 500 character summary of the issue body.")
    comment_count: int = Field(..., description="The number of comments on the issue.")
    updated_at: str = Field(..., description="The date and time the issue was last updated.")
    positive_reactions: int = Field(..., description="The number of positive reactions on the issue.")
    negative_reactions: int = Field(..., description="The number of negative reactions on the issue.")
    inferred_sentiment: str = Field(..., description="The sentiment of the issue body (one of 'positive', 'negative', or 'neutral').")
    source_repo: str = Field(..., description="The name of the repository where the issue was opened.")
    labels: List[str] = Field(..., description="A list of labels on the issue.")

def send_to_honeycomb(client, responses):
    builder = client.new_builder()
    for response in responses:
        for field, value in response.dict().items():
            if isinstance(value, list):
                for item in value:
                    builder.add_field(field, item)
            else:
                builder.add_field(field, value)
        evt = builder.new_event()
        evt.send()

def process_issue(issue):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a system that parses GitHub issues and extracts data from them."
            },
            {
                "role": "user",
                "content": issue
            }
        ],
        response_model=Issue
    )
    return resp

def run_issue_process(issues):
    responses = []
    total_tasks = len(issues)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_issue, json_content): json_content for json_content in issues}
        completed_tasks = 0

        for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks):
            try:
                response = future.result()
                responses.append(response)
            except Exception as exc:
                print(f'Generated an exception: {exc}')
            completed_tasks += 1
    return responses

def get_username_and_repository():
    if len(sys.argv) != 3:
        print("Please provide a username and repository as command line arguments.")
        sys.exit(1)
    
    username = sys.argv[1]
    repository = sys.argv[2]
    
    return username, repository

def dry_run(responses):
    print("---- DRY RUN OUTPUT ----")
    print(f"Total issues processed: {len(responses)}")
    for response in responses:
        print(f"Issue Title: {response.title}")
        print(f"URL: {response.url}")
        print(f"Summary: {response.body_summary}")
        print("-----")

def main():
    username, repository = get_username_and_repository()
    issues = github.issues_on(username, repository, state='open')
    filtered_issues = [issue for issue in issues if "renovate-bot" not in issue.user.login and "forking-renovate[bot]" not in issue.user.login]
    issues_json = [issue.as_json() for issue in filtered_issues]
    responses = run_issue_process(issues_json)
    hc = libhoney.Client(writekey=os.getenv('HONEYCOMB_API_KEY'), dataset='otel-github-issues', api_host=os.getenv('HONEYCOMB_ENDPOINT'))
    if (os.getenv('DRY_RUN')):
        dry_run(responses)
    else:
        send_to_honeycomb(hc, responses)
    hc.close()

if __name__ == "__main__":
    main()