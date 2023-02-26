import runpod
import requests
import infer
import os
import zipfile
import uuid
from urllib.parse import urlparse
from runpod.serverless.utils import download, upload, rp_cleanup
from boto3 import session

MODEL = infer.Predictor()


def download_weights_from_url(file_url):
    os.makedirs("temp", exist_ok=True)

    download_response = requests.get(file_url, timeout=30)
    download_path = urlparse(file_url).path

    original_file_name = os.path.basename(download_path)
    file_type = os.path.splitext(original_file_name)[1].replace(".", "")

    file_name = f"{uuid.uuid4()}"

    with open(f"temp/{file_name}.{file_type}", "wb") as output_file:
        output_file.write(download_response.content)

    if file_type == "zip":
        unziped_directory = "weights"
        os.makedirs(unziped_directory, exist_ok=True)
        with zipfile.ZipFile(f"temp/{file_name}.{file_type}", "r") as zip_ref:
            zip_ref.extractall(unziped_directory)
        unziped_directory = os.path.abspath(unziped_directory)
    else:
        unziped_directory = None

    return unziped_directory


def download_weights_from_s3(s3_config):
    sess = session.Session()

    original_file_name = os.path.basename(s3_config["fileName"])
    file_type = os.path.splitext(original_file_name)[1].replace(".", "")
    file_name = f"{uuid.uuid4()}"

    endpoint_url = s3_config.get(
        "endpointUrl", os.environ.get("BUCKET_ENDPOINT_URL", None)
    )
    bucket_name = s3_config.get("bucketName")
    aws_access_key_id = s3_config.get(
        "accessId", os.environ.get("BUCKET_ACCESS_KEY_ID", None)
    )
    aws_secret_access_key = s3_config.get(
        "accessSecret", os.environ.get("BUCKET_SECRET_ACCESS_KEY", None)
    )
    region_name = s3_config.get("regionName", "weur")
    client = sess.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )

    os.makedirs("temp", exist_ok=True)
    with open(f"temp/{file_name}.{file_type}", "wb") as output_file:
        print(f"bucket: {bucket_name}, file: {s3_config['fileName']}")
        client.download_fileobj(bucket_name, s3_config["fileName"], output_file)

    if file_type == "zip":
        unziped_directory = "weights"
        os.makedirs(unziped_directory, exist_ok=True)
        with zipfile.ZipFile(f"temp/{file_name}.{file_type}", "r") as zip_ref:
            zip_ref.extractall(unziped_directory)
        unziped_directory = os.path.abspath(unziped_directory)
    else:
        unziped_directory = None

    return unziped_directory


def run(job):
    """
    Job input:
    {
      "samples": [
        {
          "prompt": str,
          "negative_prompt": str,
          "guidance_scale": float,
          "inference_steps": int,
          "num_outputs": int,
          "seed": int,
        }
      ],
      "weights": {
        "download_url": str,
        "s3Config": {
          "accessId": "743d47ad447d4235b36fae020ce126fb",
          "accessSecret": "ddd7e18a695fa472a3ff9b33d173e82ac091826aae19fbcb43040cfed9286495",
          "bucketName": "exeai-dreambooth-output",
          "fileName": str,
          "endpointUrl": "https://87aa5a560c0f7dbb4281d93c1a110324.r2.cloudflarestorage.com/exeai-dreambooth-output"
         },
      },
      "s3Config": {}
    }

    """
    job_input = job["input"]
    weights = job_input["weights"]
    if weights.get("download_url") is not None:
        download_weights_from_url(weights["download_url"])
    elif weights.get("s3Config") is not None:
        download_weights_from_s3(weights["s3Config"])

    MODEL.setup()
    output_s3_config = job_input.get("s3Config")
    infer = job_input["infer"]
    if infer.get("image") is not None:
        infer["image"] = download.download_input_objects([infer["image"]])
    job_results = MODEL.predict(
        prompt=infer.get("prompt"),
        negative_prompt=infer.get("negative_prompt"),
        image=infer.get("image"),
        width=infer.get("width", 512),
        height=infer.get("height", 512),
        prompt_strength=infer.get("prompt_strength", 0.8),
        guidance_scale=infer.get("guidance_scale", 7.5),
        num_outputs=infer.get("num_outputs", 1),
        seed=infer.get("seed", 512),
        disable_safety_check=infer.get("disable_safety_check", False),
        scheduler="DDIM",
    )

    if output_s3_config is not None:
        images = upload.bucket_upload(job["id"], job_results, output_s3_config)
    else:
        images = []
        for idx, sample in enumerate(job_results):
            images.append({"image": upload.upload_image(job["id"], sample, index)})
    job_output["images"] = images
    return job_output


runpod.serverless.start({"handler": run})
