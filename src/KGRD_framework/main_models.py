
import os
import time
import json
import argparse
import os.path as osp
from tqdm import tqdm
import asyncio
from openai import OpenAI
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.agents import (
    UserProxyAgent,
    AssistantAgent,
)
import random
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from utils import *
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.base._task import *
from typing import Sequence
from collections import defaultdict
from autogen_ext.models.ollama import OllamaChatCompletionClient
from collections import Counter
import warnings
warnings.filterwarnings(
    "ignore",
    message="Could not find <think>..</think> field in model response content."
)

config = json.load(open("PATH/TO/config.json", "r"))
def parse_args():
    parser = argparse.ArgumentParser(description="Medagents Setting")
    parser.add_argument(
        "--project_name",
        type=str,
        default="KGRD",
        help="output file name",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek",
        help="the llm models",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="test dataset name",
        help="choice different dataset",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="inital",
        choices=["inital", "follow_up"],
        help="initial stage includes test and measure",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="choice different stages",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="log file",
    )
    parser.add_argument(
        "--num_doctors",
        type=int,
        default=6,
        help="number of experts, >3, including tools",
    )
    parser.add_argument("--n_round", type=int, default=15, help="attempt_vote")
    parser.add_argument("--withtool", type=bool, default=True, help="with tool or not")
    parser.add_argument(
        "--SelectTool", type=str, default=None, help="Used tools name.split(',')"
    )


    args = parser.parse_args()

    return args


async def process_single_case(args, dataset, idx, output_dir, model_client):

    case_cost = 0.0
    case_info = {}
    (case_type, case_name, case_crl,
                case_initial_presentation, case_follow_up_presentation,
                HPO_LIST, HPO_NAME_LIST, GENE, DISEASE_FROM_GENE,
                DISEASE_FROM_PHENOTYPE, DISEASE_FROM_PATIENT_LIKE_ME)=dataset[idx]
    KnowledgeVerifier_result_ok = None
    KnowledgeVerifier_result_exp = None
    
    json_name = f"{case_crl}.json"
    identify = f"{args.num_doctors}-{args.n_round}"
    output_dir = osp.join(
        output_dir,
        args.project_name,
        args.stage,
        args.model_name,
        identify,
        str(args.times),
    )
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    file_names = os.listdir(output_dir)
    json_files = [file for file in file_names if file.endswith(".json")]

    if json_name in json_files:
        return f"{json_name} done."
    else:
        print('[Case Crl] ',case_crl)

    if args.stage == "inital":
        case_presentation = case_initial_presentation
    elif args.stage == "follow_up":
        case_presentation = case_follow_up_presentation
    else:
        raise NotImplementedError
    

    reasoning_models = ["deepseek-r1:32b", "deepseek-reasoner"]
    reasoning = args.model_name in reasoning_models
    ollama_deepseek = ["deepseek-r1:32b"]
    ollama_deepseek = args.model_name in ollama_deepseek
    if reasoning:
        if ollama_deepseek:
            model_context = OllamaReasoningModelContext()
        else:
            model_context = ReasoningModelContext()
    else:
        model_context = None
    
    

    Docs = []
    if args.withtool and args.SelectTool:
        SelectTools = args.SelectTool.split(",")
    tool_agents = []

    def create_tool_agent(
        name, tool_func, description=None, necessary_result=None, system_msg=None,model_context = None
    ):
        if name in ['PhenoDMiner','GeneDPredictor','PatientDMatcher']:
            model_context = BeforeToolModelContext()

        else:
            model_context = None

        tools = None if necessary_result else tool_func
        return AssistantAgent(
            name=name,
            model_client=model_client,
            description=description,
            system_message=system_msg,
            tools=tools,
            model_context=model_context, 

        )
    if "PhenoDMiner" in SelectTools:
        system_msg = PhenoDMiner_agent_system_message(
            hpo_name_list = HPO_NAME_LIST,
            disease_from_phenotype = DISEASE_FROM_PHENOTYPE if DISEASE_FROM_PHENOTYPE else None,
        )
        PhenoDMiner_agent = create_tool_agent(
            "PhenoDMiner",
            [PhenoDMiner_tool_call],
            description=""" Infers likely diseases from phenotype profiles via ontology-based associations""",
            necessary_result=DISEASE_FROM_PHENOTYPE if DISEASE_FROM_PHENOTYPE else None,
            system_msg=system_msg,
        )
        tool_agents.append(PhenoDMiner_agent)

    if "GeneDPredictor" in SelectTools and GENE:
        system_msg = GeneDPredictor_agent_system_message(
                hpo_name_list = HPO_NAME_LIST,
                disease_from_gene = DISEASE_FROM_GENE if DISEASE_FROM_GENE else None,
                Gene=GENE
        )
        GeneDPredictor_agent = create_tool_agent(
            "GeneDPredictor",
            [g2d,query_one_hop_gene_disease],
            description="""Predicts disease links for a gene using integrated rare disease knowledge graph. """,
            necessary_result=DISEASE_FROM_GENE if DISEASE_FROM_GENE else None,
            system_msg=system_msg,
        )
        tool_agents.append(GeneDPredictor_agent)


    if "PatientDMatcher" in SelectTools:

        system_msg = PatientDMatcher_agent_system_message(
            hpo_name_list = HPO_NAME_LIST,
            disease_from_patient_like_me=
                DISEASE_FROM_PATIENT_LIKE_ME if DISEASE_FROM_PATIENT_LIKE_ME else None,
        )
        PatientDMatcher_agent = create_tool_agent(
            "PatientDMatcher",
            [sapbert_d_patient], 
            description=""" - Finds similar past patient cases to support diagnosis.""",
            necessary_result=DISEASE_FROM_PATIENT_LIKE_ME if DISEASE_FROM_PATIENT_LIKE_ME else None,
            system_msg=system_msg,
        )
        tool_agents.append(PatientDMatcher_agent)

    if "KnowledgeVerifier" in SelectTools:
        system_msg = KnowledgeVerifier_agent_system_message(
            case_initial_presentation=case_initial_presentation,
            HPO_NAME_LIST=HPO_NAME_LIST,
            GENE=GENE,
            HPO_LIST=HPO_LIST,
            )
        KnowledgeVerifier_agent = create_tool_agent(
            "KnowledgeVerifier",
            tool_func=[CaseInput, verify],
            description=(
                "Verifies whether the final most-likely diagnosis is sufficiently supported "
                "by multi-source clinical evidence (KG/DB/gene/phenotype/literature), "
                "surfacing a transparent, module-wise explanation."
            ),
            system_msg=system_msg,
        )
        tool_agents.append(KnowledgeVerifier_agent)


    

    if args.num_doctors - len(SelectTools) <= 0:
        raise NotImplementedError


    



    num_doctors = args.num_doctors
    if args.withtool:
        num_doctors = args.num_doctors - len(SelectTools)

    consultant_message = get_consultant_message(case_presentation, int(num_doctors))

    response = justchat(consultant_message)
    print(response)

    top_k_specialists = parse_json(response)["top_k_specialists"]
    assert len(top_k_specialists) == int(num_doctors)


    file_path = "/share/gguilin/rd-project/Multi-agent-conversation-for-disease-diagnosis/utils/all_expert_prompt.jsonl"

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))

    expert_prompt_dict = {list(d.keys())[0]: list(d.values())[0] for d in data}
    aftertoolcontext = afterToolsjsonContext()

    for specialist in top_k_specialists:
        name = to_valid_identifier(specialist)
        doc_system_message = get_doc_system_message(
            doctor_name=expert_prompt_dict.get(name, "base_doctor"), stage=args.stage, gene = GENE
        )

        Doc = AssistantAgent(
            name=name,
            model_client=model_client,
            system_message=doc_system_message,
            model_context=aftertoolcontext,

        )
        Docs.append(Doc)

    termination = TextMentionTermination("TERMINATE")

    supervisor_system_message = get_supervisor_system_message(
        specialists=','.join([i.name for i in Docs]),tools=','.join([i.name for i in tool_agents]),
    )

    Supervisor = AssistantAgent(
        name="Supervisor",
        model_client=model_client,
        system_message=supervisor_system_message,
        model_context=model_context,

    )

    inital_message = get_inital_message(
        patient_history=case_presentation, stage=args.stage
    )

    Init_patient_info = AssistantAgent(
        name="Patient_info",
        model_client=model_client,
        system_message=inital_message,
        model_context=model_context,
       
    )

    agents = [Init_patient_info] + Docs + tool_agents + [Supervisor]
    print("Doctors and Tools used in this case: " + str([i.name for i in agents]))
    
    @simple_retry(max_attempts=3, delay=2)  # 设置重试次数和间隔
    def selector_func(
        messages: Sequence[BaseAgentEvent | BaseChatMessage],
    ) -> str | None:
        if not messages:
            return Init_patient_info.name

        valid_messages = [
            m for m in messages
            if isinstance(m, (BaseChatMessage, ToolCallSummaryMessage)) and len(m.content) > 1
        ]
        if not valid_messages:
            return Init_patient_info.name

        last_message = valid_messages[-1]

        patient_name = Init_patient_info.name
        supervisor_name = Supervisor.name
        
        docs_names = [agent.name for agent in Docs]
        if args.withtool:
            tools_name = [agent.name for agent in tool_agents]
        else:
            tools_name = []
        if not any(msg.source == patient_name for msg in messages):
            return patient_name
        if last_message.source == supervisor_name:
            return None

        if "KnowledgeVerifier" in tools_name:
            Verifier = 'KnowledgeVerifier'
        else:
            Verifier = None
        

        if last_message.source == patient_name and "PhenoDMiner" in tools_name:
            return "PhenoDMiner"
        if last_message.source == "PhenoDMiner" and "PatientDMatcher" in tools_name and "PatientDMatcher" not in [i.source for i in valid_messages]:
            return "PatientDMatcher"
        if last_message.source == "PatientDMatcher" and "GeneDPredictor" in tools_name and "GeneDPredictor" not in [i.source for i in valid_messages]:
            return "GeneDPredictor"
        if last_message.source =="KnowledgeVerifier" and isinstance(last_message, AssistantMessage):
            try:
                KnowledgeVerifier_result = parse_json(last_message.content)

                if isinstance(KnowledgeVerifier_result,dict):
                    KnowledgeVerifier_result_ok = KnowledgeVerifier_result.get('ok',None)
                    KnowledgeVerifier_result_exp = KnowledgeVerifier_result.get('explanation',None)
                elif isinstance(KnowledgeVerifier_result,tuple):
                    KnowledgeVerifier_result_ok,KnowledgeVerifier_result_exp = KnowledgeVerifier_result
                

                if KnowledgeVerifier_result_ok:
                    return supervisor_name
            except Exception as e:
                print(f"KnowledgeVerifier_result error: {e}")
                pass




        doc_msg_counts = Counter(msg.source for msg in messages if msg.source in docs_names)

        total_rounds = int(args.n_round)
        total_doc_msgs = sum(doc_msg_counts.values())

        if len(valid_messages) >= total_rounds - 1:
            return supervisor_name

        try:
            doctor_messages = [
                m
                for m in messages
                if isinstance(m, BaseChatMessage) and m.source in docs_names
            ]
            doctor_speakers = set(m.source for m in doctor_messages)

            if len(doctor_speakers) < max(4, len(docs_names)//2):
                # too few doctors spoke yet
                pass
            else:

                last_messages_by_doctor = {}

                for m in reversed(doctor_messages):
                    if m.source not in last_messages_by_doctor:
                        last_messages_by_doctor[m.source] = m

                last_messages = list(last_messages_by_doctor.values())

                if len(last_messages) <= len(docs_names):
                    pattern = r"Most Likely Diagnosis\*\*[\s\S]*?\*\*(.*?)\*\*"
                    texts = []
                    for text in [m.content.strip() for m in last_messages]:
                        match = re.search(pattern, text, re.DOTALL)
                        if match:
                            mostlikelydiagnosis = match.group(1).strip()
                        else:
                            mostlikelydiagnosis=justchat('extract the most Likely Diagnosis, only output the disease name, not other context :'+text)
                        texts.append(mostlikelydiagnosis)
                    embeddings = np.stack([get_embedding(t) for t in texts])
                    n_clusters = 3
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
                    labels = kmeans.labels_
                    counts = Counter(labels)
                    total = len(labels)
                    entropy = 0.0
                    for freq in counts.values():
                        p = freq / total
                        entropy -= p * np.log2(p)

                    max_entropy = np.log2(n_clusters)
                    entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0
                    

                    print(f"[DEBUG] entropy={entropy:.4f}, ratio={entropy_ratio:.2%}")

                    if entropy_ratio <= 0.75:
                        print(
                            "Entropy low, experts are consistent. Let supervisor summarize."
                        )
                        return supervisor_name
                    else: 
                        if "KnowledgeVerifier" not in [i.source for i in valid_messages] and "KnowledgeVerifier" in tools_name:
                            return "KnowledgeVerifier"
                        
        except Exception as e:
            print(f"[WARNING] Entropy computation failed: {e}")

        return retry_llm_selector(
                last_message,
                doc_msg_counts,
                docs_names,
                Verifier,
                supervisor_name,
                agents,
                tools_name,
            )

            

    manager = SelectorGroupChat(
        agents,
        model_client=model_client,
        termination_condition=termination,
        max_turns=args.n_round,
        selector_func=selector_func,
        max_selector_attempts=args.n_round // (args.num_doctors) + 3,
        model_context=model_context,
    )

    chat_history = []
    async for message in manager.run_stream(
        task="Multi doctor discuss to diagnosis disease"
    ):

        chat_history.append(message)
        print(message)  



    for agent in agents:
        case_cost += 1.0

    critic_output = []
    for item in chat_history:
        try:
            if isinstance(item, BaseChatMessage):
                if (
                    item.source == "Supervisor"
                    and '"Most Likely Diagnosis":' in item.content
                ):
                    critic_output.append(item.content)
        except:
            pass

        try:
            if isinstance(item, BaseChatMessage):
                if item.source == "KnowledgeVerifier":
                    KnowledgeVerifier_result = parse_json(item.content)
                    if isinstance(KnowledgeVerifier_result,dict):
                        KnowledgeVerifier_result_ok = KnowledgeVerifier_result.get('ok',None)
                        KnowledgeVerifier_result_exp = KnowledgeVerifier_result.get('explanation',None)
                    elif isinstance(KnowledgeVerifier_result,tuple):
                        KnowledgeVerifier_result_ok,KnowledgeVerifier_result_exp = KnowledgeVerifier_result

        except:
            pass

    if critic_output:
        try:
            syn_report = critic_output[-1]
            json_output = parse_json(syn_report)
        except:
            syn_report = critic_output[-2]
            json_output = parse_json(syn_report)

        case_info["Type"] = case_type
        case_info["Crl"] = case_crl
        case_info["Cost"] = case_cost
        case_info["Presentation"] = case_presentation
        case_info["Name"] = case_name
        case_info["Most Likely"] = json_output.get("Most Likely Diagnosis")
        case_info["Other Possible"] = json_output.get(
            "Differential"
        ) or json_output.get("Differential Diagnosis") or json_output.get("Differential Diagnoses") or json_output.get("Other Possible")

        case_info['KnowledgeVerifier_ok'] = KnowledgeVerifier_result_ok
        case_info['KnowledgeVerifier_exp'] = KnowledgeVerifier_result_exp


        if args.stage == "inital":
            case_info["Recommend Tests"] = json_output.get(
                "Recommend Tests"
            ) or json_output.get("Recommended Tests")

        recorder_path = osp.join(output_dir, json_name)
        with open(recorder_path, "w") as file:
            json.dump(case_info, file, indent=4)


import traceback


async def main_async(args):
    dataset = MedDataset(dataname=args.dataset_name)
    data_len = len(dataset)
    output_dir = args.output_dir
 
    if args.model_name in ["deepseek"]:
        model_client = OpenAIChatCompletionClient(
            model="deepseek-chat",
            base_url="https://",    
            api_key="YOUR_API_KEY", 

            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.R1, #NOT THINKING IN KGRD, JUST USE THE PARAMETER SETTING OF ModelFamily.R1
                "structured_output": True,
            },
            temperature=0.3,
            max_tokens=4096
        )

        for idx in tqdm(range(data_len)):
            try:
                await process_single_case(args, dataset, idx, output_dir, model_client)
            except Exception as e:
                print(f"Failed to process case {idx} after all attempts: {str(e)}")
                continue
        # YOU CAN ADD MORE MODELS AS model_client



if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
