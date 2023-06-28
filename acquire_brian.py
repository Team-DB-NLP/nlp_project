# imports

import os
import re
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

import acquire_brian as acq

import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd

from pprint import pprint 

# functions

REPOS = ['Project-MONAI/MONAI',
 'GoogleCloudPlatform/healthcare',
 'kakoni/awesome-healthcare',
 'wanghaisheng/healthcaredatastandard',
 'HealthCatalyst/healthcareai-py',
 'microsoft/HealthBotContainerSample',
 'isaacmg/healthcare_ml',
 'TheAlphamerc/flutter_healthcare_app',
 'nextgenhealthcare/connect',
 'OCA/vertical-medical',
 'HealthCatalyst/healthcareai-r',
 'acoravos/healthcare-blockchains',
 'itachi9604/healthcare-chatbot',
 'prasadseemakurthi/Deep-Neural-Networks-HealthCare',
 'pratik008/HealthCare_Twitter_Analysis',
 'MoH-Malaysia/covid19-public',
 'DataKind-SG/healthcare_ASEAN',
 'llSourcell/AI_for_healthcare',
 'sunlabuiuc/PyHealth',
 'llSourcell/How_to_Build_a_healthcare_startup',
 'GoogleCloudPlatform/healthcare-data-harmonization',
 'vsharathchandra/AI-Healthcare-chatbot',
 'Rishabh42/HealthCare-Insurance-Ethereum',
 'PacktPublishing/Machine-Learning-for-Healthcare-Analytics-Projects',
 'GoogleCloudPlatform/healthcare-dicom-dicomweb-adapter',
 'newmediamedicine/CollaboRhythm',
 'neee/healthcare-service',
 'openhealthcare/opal',
 'SoumyaRSethi/Data-Science-Capstone-Healthcare',
 'loutfialiluch/HealthCare',
 'pramodramdas/digital_healthcare',
 'grfiv/healthcare_twitter_analysis',
 'instamed/healthcare-payments-blockchain',
 'sarveshraj/blockchain-for-healthcare',
 'STRML/Healthcare.gov-Marketplace',
 'medtorch/awesome-healthcare-ai',
 'microsoft/InnerEye-DeepLearning',
 'Project-Based-Learning-IT/healthcare-appointment-scheduling-app',
 'edaaydinea/AI-Projects-for-Healthcare',
 'IBMStreams/streamsx.health',
 'vanderschaarlab/mlforhealthlabpub',
 'MichaelAllen1966/1804_python_healthcare',
 'IBM-MIL/IBM-Ready-App-for-Healthcare',
 'qgzang/ComputationalHealthcare',
 'microsoft/healthcare-shared-components',
 'bluehalo/node-fhir-server-core',
 'CMSgov/HealthCare.gov-Styleguide',
 'medplum/medplum',
 'nickls/awesome-healthcare-datasets',
 'AileenNielsen/OReillyHealthcareData',
 'Conservatory/healthcare.gov-2013-10-01',
 'informatici/openhospital',
 'technext/HealthCare',
 'coronasafe/care',
 'microsoft/healthcare-apis-samples',
 'GoogleCloudPlatform/healthcare-deid',
 'coronasafe/care_fe',
 'openboxes/openboxes',
 'hyperledger-labs/sawtooth-healthcare',
 'chvlyl/ML_in_Biomed',
 'abuanwar072/Production-Ready-Doctor-Consultant-App-UI-',
 'amanjeetsahu/AI-for-Healthcare-Nanodegree',
 'rahulremanan/HIMA',
 'IMA-WorldHealth/bhima',
 'GoogleCloudPlatform/healthcare-data-harmonization-dataflow',
 'CodeForBaltimore/Healthcare-Rollcall',
 'sample2025nit/HealthCareEx',
 'aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks',
 'PacktPublishing/Healthcare-Analytics-Made-Simple',
 'simpledotorg/simple-android',
 'katalon-studio-samples/healthcare-tests',
 'pras75299/Healthcare-Website',
 'rajagopal28/healthcare-server',
 'blencorp/HealthCare.gov-Open-Source-Release',
 'Jianing-Qiu/Awesome-Healthcare-Foundation-Models',
 'metriport/metriport',
 'IBM/Medical-Blockchain',
 'mp2893/retain',
 'ESS-LLP/smarte',
 'PacktPublishing/Applied-Machine-Learning-For-Healthcare',
 'informatici/openhospital-core',
 'opensource-emr/hospital-management-emr',
 'clinical-meteor/meteor-on-fhir',
 'openmrs/openmrs-core',
 'digital-asset/ex-healthcare-claims-processing',
 'Ivanzgj/HealthCare',
 'TheDesignMedium/healthcare-website',
 'cevheri/healthcare',
 'wso2/open-healthcare-docs',
 'OmRajpurkar/Healthcare-Chatbot',
 'AmitXShukla/Healthcare-Management-App-Flutter_Firebase',
 'IRCAD/sight',
 'hasyed/HealthCareApp',
 'Qingbao/HealthCareStepCounter',
 'arvindsis11/Ai-Healthcare-Chatbot',
 'microsoft/Healthcare-Blockchain-Solution-Accelerator',
 'openmrs/openmrs-module-radiology',
 'VectorInstitute/cyclops',
 'hapifhir/hapi-fhir',
 'aws-samples/amazon-sagemaker-healthcare-fraud-detection']
 

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(get_readme_download_url(contents)).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)