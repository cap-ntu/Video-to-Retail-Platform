import Sidebar from "../common/Sidebar";
import Switch from "react-router/Switch";
import Route from "../../../routes/RouteWrapper";
import Redirect from "../../../routes/RedirectWrapper";
import React from "react";
import {styles} from "../common/layout";
import ModelManagement from ".";
import ModelDetails from "./ContainerModelDetails";
import withStyles from "@material-ui/core/styles/withStyles";
import ModelNew from "./ContainerModelNew";
import MessageBars from "./MessageBars";

const ModelRouter = ({classes, match}) => (
    <div className={classes.root}>
        <Sidebar currentPath={match.path}/>
        <div className={classes.grow}>
            <Switch>
                {Route({exact: true, path: '/all', component: ModelManagement})}
                {Route({exact: true, path: '/search', component: ModelDetails})}
                {Route({exact: true, path: '/upload', component: ModelNew})}
                {Redirect({from: '/', to: '/all'})}
                {/*<Redirect to={'/404'}/>*/}
            </Switch>
            <MessageBars/>
        </div>
    </div>
);

export default withStyles(styles)(ModelRouter);
