import React from 'react';
import Drawer from "@material-ui/core/Drawer/Drawer";
import CssBaseline from "@material-ui/core/CssBaseline/CssBaseline";
import PropTypes from "prop-types";
import withStyles from "@material-ui/core/styles/withStyles";
import HeaderPlaceHolder from "./HeaderPlaceholder";

const styles = {
    drawer: {
        flexShrink: 0,
    },
    drawerRoot: {
        border: 0,
    },
    titleRoot: {
        paddingTop: 0,
        paddingBottom: 0,
    },
    drawerPaper: {},
    chart: {
        height: '8rem',
        marginBottom: '1rem',
    }
};

const RightNavigation = ({classes, children}) => (
    <Drawer variant="permanent"
            anchor={"right"}
            PaperProps={{classes: {root: classes.drawerRoot}}}
            className={classes.drawer}
            classes={{paper: classes.drawerPaper}}>
        <HeaderPlaceHolder/>
        <CssBaseline/>
        {children}
    </Drawer>
);

RightNavigation.propTypes = {
    classes: PropTypes.object.isRequired,
    children: PropTypes.node
};

export default withStyles(styles)(RightNavigation);
