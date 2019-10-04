import React from 'react';
import withStyles from "@material-ui/core/styles/withStyles";
import * as PropTypes from "prop-types";
import ResourceManagement from "./resource";
import Sidebar from "./common/Sidebar";
import {styles} from "./common/layout";

const DashboardApp = ({classes, match}) => (
    <div className={classes.root}>
        <Sidebar currentPath={match.path}/>
        <div className={classes.grow}>
            <ResourceManagement/>
        </div>
    </div>
);

DashboardApp.propTyps = {
    classes: PropTypes.object.isRequired,
    CPU: PropTypes.arrayOf(
        PropTypes.object.isRequired,
    ).isRequired,
};

export default withStyles(styles)(DashboardApp);
