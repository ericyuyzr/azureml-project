import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute, NetworkSettings
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

# 1. CONNECT TO AZURE ML
SUBSCRIPTION_ID = "47750dae-d97d-4bc8-a768-0bb7f35fb08c"
RESOURCE_GROUP = "learn-networking-rg"
WORKSPACE_NAME = "azml2"
COMPUTE_NAME = "aml-cluster-private"

# -------------------------------------------------------------------
# Connect to Azure ML Workspace
# -------------------------------------------------------------------
ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

# -------------------------------------------------------------------
# Create or Get Compute Cluster
# -------------------------------------------------------------------
try:
    compute = ml_client.compute.get(COMPUTE_NAME)
    print(f"Compute '{COMPUTE_NAME}' already exists.")

except ResourceNotFoundError:
    print(f"Creating compute '{COMPUTE_NAME}'...")

    cpu_compute = AmlCompute(
        name=COMPUTE_NAME,
        size="Standard_E4ds_v4",
        min_instances=0,
        max_instances=2
    )

    compute = ml_client.compute.begin_create_or_update(cpu_compute).result()
    print("Compute created successfully.")

# -------------------------------------------------------------------
# Training Script
# -------------------------------------------------------------------
training_script = f"""
import os
import pickle

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


print("Training model...")

X, y = make_classification(
    n_samples=100,
    n_features=4,
    random_state=42,
)

model = LogisticRegression()
model.fit(X, y)

# Save model to AzureML outputs directory
output_dir = os.environ.get("AZUREML_OUTPUTS_DIR", ".")

model_path = os.path.join(output_dir, "toy_sklearn_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {{model_path}}")


print("Model registered successfully.")
"""

with open("train_vnet.py", "w") as f:
    f.write(training_script)

# -------------------------------------------------------------------
# Submit Job
# -------------------------------------------------------------------
job = command(
    code=".",
    command="python train_vnet.py",
    compute=COMPUTE_NAME,
    environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
    environment_variables={
        "SUBSCRIPTION_ID": SUBSCRIPTION_ID,
        "RESOURCE_GROUP": RESOURCE_GROUP,
        "WORKSPACE_NAME": WORKSPACE_NAME,
    },
)

print("Submitting job...")

returned_job = ml_client.jobs.create_or_update(job)

print("Job submitted!")
print(returned_job.studio_url)

os.remove("train_vnet.py")