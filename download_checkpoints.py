import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    '--region',
    choices=['us-phoenix-1', 'uk-london-1', 'us-ashburn-1'],
    default='us-phoenix-1',
)
parser.add_argument(
    '--local',
    action='store_true',
    default=False,
    help="Use this flag when running outside of an OCI instance.",
)
args = parser.parse_args()


def download(object_name, save_name):
  if os.path.exists(save_name):
    print(
        "The requested checkpoint has already been downloaded to the following location: ",
        save_name
    )
  else:
    oci_command = (
        'oci os object get '
        '--namespace idh41mwrww9e '
        '--bucket-name checkpoint-storage '
        f'--name \'{object_name}\' '
        f'--file \'{save_name}\' '
        f'--region {args.region} '
    )
    if not args.local:
      oci_command += '--auth instance_principal'

    subprocess.run(
        oci_command,
        shell=True,
        check=True,
    )


object_name = 'wonder3d-ckpts.tar.gz'
save_name = 'wonder3d-ckpts.tar.gz'
if not os.path.exists(save_name):
    print(f'Downloading Wonder3D checkpoint to `{save_name}`')
    download(object_name, save_name)

# Now untar it
if not os.path.exists('ckpts/unet'):
    os.system('tar -zxvf wonder3d-ckpts.tar.gz')