from azure.ai.ml import MLClient
from azure.ai.ml import load_job
from azure.ai.ml.entities import (
    JobSchedule,
    RecurrenceTrigger,
    RecurrencePattern,
)
from azure.identity import DefaultAzureCredential
import os

subscription_id = os.environ["SUBSCRIPTION_ID"]
resource_group = os.environ["RESOURCE_GROUP"]
workspace_name = os.environ["WORKSPACE_NAME"]

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name,
)

print("Loading pipeline job...")

pipeline_job = load_job(
    source="./azureml/pipeline.yml"
)
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job
)

print("Creating schedule...")

schedule = JobSchedule(
    name="toy-daily-schedule",
    display_name="Toy Daily Batch Schedule",
    trigger=RecurrenceTrigger(
        frequency="day",
        interval=1,
        schedule=RecurrencePattern(
            hours=[10],
            minutes=[0],
        ),
        time_zone="Pacific Standard Time",
    ),
    create_job=pipeline_job,
)

ml_client.schedules.begin_create_or_update(schedule).result()

print("Schedule deployed successfully!")