import Route from "../../../routes/RouteWrapper";
import React from "react";
import Switch from "react-router/Switch";
import withStyles from "@material-ui/core/styles/withStyles";
import VideoManagement from ".";
import UploadCard from "./ContainerUploadCard";
import Sidebar from "../common/Sidebar";
import {styles} from "../common/layout";
import MessageBar from "./MessageBars";

const VideoRouter = ({classes, match}) => (
    <div className={classes.root}>
        <Sidebar currentPath={match.path}/>
        <div className={classes.grow}>
            <Switch>
                {Route({exact: true, path: '/', component: VideoManagement})}
                {Route({exact: true, path: '/upload', component: UploadCard})}
                {/*<Redirect to={'/404'}/>*/}
            </Switch>
            <MessageBar/>
        </div>
    </div>
);

export default withStyles(styles)(VideoRouter);
