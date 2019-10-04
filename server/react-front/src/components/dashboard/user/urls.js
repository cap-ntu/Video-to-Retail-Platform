import React from "react";
import {styles} from "../common/layout";
import UserManagement from ".";
import Sidebar from "../common/Sidebar";
import Switch from "react-router/Switch";
import Route from "../../../routes/RouteWrapper";
import withStyles from "@material-ui/core/styles/withStyles";
import MessageBars from "./MessageBars";

const UserRouter = ({classes, match}) => (
    <div className={classes.root}>
        <Sidebar currentPath={match.path}/>
        <div className={classes.grow}>
            <Switch>
                {Route({exact: true, path: '/', component: UserManagement})}
                {/*<Redirect to={'/404'}/>*/}
            </Switch>
            <MessageBars/>
        </div>
    </div>
);

export default withStyles(styles)(UserRouter);
