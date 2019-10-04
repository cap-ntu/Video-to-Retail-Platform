import React from "react";
import PropTypes from "prop-types";
import HomeIcon from "@material-ui/icons/HomeRounded";
import VideoIcon from "@material-ui/icons/VideoLibraryRounded";
import PeopleIcon from "@material-ui/icons/PeopleRounded";
import ViewModuleIcon from "@material-ui/icons/ViewModuleRounded";
import OSidebar from "../../common/app/Sidebar";
import withStyles from "@material-ui/core/styles/withStyles";
import Icon from "@material-ui/core/Icon";

const sidebarWidth = "240px";

const styles = {
    sidebar: {
        width: `${sidebarWidth}`,
    },
    sidebarPaper: {
        width: `${sidebarWidth}`,
    }
};

const Sidebar = ({classes, currentPath}) => (
    <OSidebar classes={{root: classes.sidebar, drawerPaper: classes.sidebarPaper}}
              sidebarList={[
                  {name: "Home", component: <HomeIcon/>, url: '/dashboard/'},
                  {name: "Video", component: <VideoIcon/>, url: '/dashboard/video'},
                  {name: "User", component: <PeopleIcon/>, url: '/dashboard/user'},
                  {name: "Model", component: <ViewModuleIcon/>, url: '/dashboard/model'},
                  {name: "Product Search", component: <Icon className={"fab fa-buysellads"}/>, url: '/dashboard/ad'},
              ]}
              currentPath={currentPath}/>
);

Sidebar.propTypes = {
    classes: PropTypes.object.isRequired,
    currentPath: PropTypes.string.isRequired,
};

export default withStyles(styles)(Sidebar);
