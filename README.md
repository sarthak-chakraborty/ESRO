# ESRO: Experience Assisted Service Reliability against Outages

This is the official repository corresponding to the paper titled **"ESRO: Experience Assisted Service Reliability against Outages"** accepted at the 2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE 2023), Kirchberg, Luxembourg.

```
@INPROCEEDINGS{10298416,
  author={Chakraborty, Sarthak and Agarwal, Shubham and Garg, Shaddy and Sethia, Abhimanyu and Pandey, Udit Narayan and Aggarwal, Videh and Saini, Shiv},
  booktitle={2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE)}, 
  title={ESRO: Experience Assisted Service Reliability against Outages}, 
  year={2023},
  volume={},
  number={},
  pages={255-267},
  doi={10.1109/ASE56229.2023.00131}}
```


## Abstract

Modern cloud services are prone to failures due to their complex architecture, making diagnosis a critical process. Site Reliability Engineers (SREs) spend hours leveraging multiple sources of data, including the alerts, error logs, and domain expertise through past experiences to locate the root cause(s). These experiences are documented as natural language text in outage reports for previous outages. However, utilizing the raw yet rich unstructured information in the reports systematically is time-consuming. Structured information, on the other hand, such as alerts that are often used during fault diagnosis, is voluminous and requires expert knowledge to discern. Several strategies have been proposed to use each source of data separately for root cause analysis. In this work, we build a diagnostic service called ESRO that recommends root causes and remediation for failures by utilizing structured as well as unstructured sources of data systematically. ESRO constructs a causal graph using alerts and a knowledge graph using outage reports, and merges them in a novel way to form a unified graph during training. A retrieval-based mechanism is then used to search the unified graph and rank the likely root causes and remediation techniques based on the alerts fired during an outage at inference time. Not only the individual alerts, but their respective importance in predicting an outage group, is taken into account during recommendation. We evaluated our model on several cloud service outages of a large SaaS enterprise over the course of ~2 years, and obtained an average improvement of 27% in rouge scores after comparing the likely root causes against the ground truth over state-of-the-art baselines. We further establish the effectiveness of ESRO through qualitative analysis on multiple real outage examples.

## Data
Data is propreitary and so it cannot be shared. The different types of data that is required in this work are:
1. A json of outages which includes the start time, stop time, symptoms, root causes and remediations,etc. as keys. (cso_anonymised.json)
2. A list of alerts, created_at timestamp and the affected services in a csv format. (alert_dummy_dataframe.csv)
3. Distinct to Alert id mapping in csv format

We have published a dummy data that show the list of outages, the list of alerts and a unique id mapping of alerts to a number in `Dummy_Data` folder.

N.B.: Make appropriate changes to the dictionary key names in the code based on your data, if needed. 

## Folder Structure
```
1. 1_make_causal_graph - Builds the causal graph from the alerts
2. 2_make_knowledge_graph - Builds a knowledge graph where the nodes are symptoms, root causes and the remediations of the outages. It also clusters the outages.
3. 3_merge_CG_KG - Combines the causal graph and the knowledge graph constructed above using timestamped mapping. It also learns a random forest model to predict the outage cluster given the one hot mapping of alerts that are being fired
4. 4_model_inference_eval - Evaluates the CK graph model based on various inference methodologies.
5. Baseline/Incident_Search - Builds the Incident Search baseline with FAISS and evaluates the performance
```

**Note**: GCN baseline is still in construction