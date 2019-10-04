# Desc: Deploy hysia model to clipper.
# Author: Zhou Shengsheng
# Date: 19-02-19

from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.python import deploy_python_closure
import subprocess
import argparse


# This function is only for the deployment, and will be replaced by the one in deploy_clipper_mmdet_container.py
def predict(inputs):
    return inputs

def replaceDefaultEntry():
    """Copy customized python_closure_container.py to model container and restart the container."""
    print('Copy python_closure_container.py to {}:{} model container'.format(model_name, model_version))
    cmd = "docker ps | grep %s:%s | awk '{print $1}'" % (model_name, model_version)
    container_id = subprocess.getoutput(cmd)
    cmd = "docker cp custom-container-entry/{}-python_closure_container.py {}:/container/python_closure_container.py"\
          .format(model_name, container_id)
    print(subprocess.check_output(cmd.split()))
    cmd = "docker restart {}".format(container_id)
    print(subprocess.check_output(cmd.split()))

def deployModelToClipper():
    """Deploy model to clipper and replace its entry."""
    global app_name, model_name, model_version

    print('Deploying model to clipper, model_name={}, model_version={}'.format(model_name, model_version))

    # Setup clipper and deploy model
    clipper_conn = ClipperConnection(DockerContainerManager(redis_port=6380))
    try:
        clipper_conn.start_clipper()
    except:
        clipper_conn.connect()
    try:
        # input_type must be bytes as inputs will be serialized into bytes with pickle
        clipper_conn.register_application(name=app_name, input_type="bytes", default_output="-1.0", slo_micros=1000000)
    except Exception as e:
        print(e)
    try:
        deploy_python_closure(
            clipper_conn,
            name=model_name,
            version=model_version,
            input_type="bytes",
            batch_size=1,
            func=predict,
            base_image='hysia-clipper-base-container-gpu')
    except Exception as e:
        print(e)
    try:
        clipper_conn.link_model_to_app(app_name=app_name, model_name=model_name)
    except Exception as e:
        print(e)

    replaceDefaultEntry()
    print('{} deployed to clipper!'.format(model_name))


def parseArgs():
    """Parse command line args."""
    parser = argparse.ArgumentParser(description="Deploy hysia model to clipper.")
    parser.add_argument("--model_name", type=str, default="mmdet", help="Model to deploy.")
    parser.add_argument("--model_version", type=str, default="1", help="Model version.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArgs()
    model_name = args.model_name
    model_version = args.model_version
    app_name = 'clipper-' + model_name

    deployModelToClipper()
