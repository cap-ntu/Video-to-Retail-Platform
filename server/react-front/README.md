# Hysia Dashboard Front End
Hysia front end is build based on React.Js with Material Design.  

# Install npm and all dependencies
If you haven't install dependencies, please run `npm install`. After all dependencies are install, run `npm run start`

# How to deploy in Django
1. Go to [`react-build`](https://github.com/HuaizhengZhang/Hysia/tree/master/hysia/server/react-build)
2. run `./build.sh index.html`, this automatically move all static files to `DjangoRoot/static/`
and `index.html` to `Django/templates/`. If you have any special request, please refer to `build.sh` for modification.
