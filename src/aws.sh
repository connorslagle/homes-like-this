# set up for aws
# activate ML source
activate source tensorflow_p36

# init git repo
cd ~
git clone https://github.com/connorslagle/homes-like-this.git
git checkout dev

# make req dirs
mkdir -p ~/homes-like-this/data/proc_imgs/128/color
mkdir -p ~/homes-like-this/data/proc_imgs/128/gray
mkdir -p ~/homes-like-this/data/Xs
mkdir -p ~/homes-like-this/data/ys
mkdir -p ~/homes-like-this/models


# load imgs from s3
export AWS_PROFILE='connor_iam'
aws s3 cp --recursive s3://homes-like-this/proc_imgs/128/color ~/homes-like-this/data/proc_imgs/128/color
aws s3 cp --recursive s3://homes-like-this/proc_imgs/128/gray ~/homes-like-this/data/proc_imgs/128/gray

