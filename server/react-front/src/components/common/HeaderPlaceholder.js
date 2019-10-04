import React from "react";
import withStyles from "@material-ui/core/styles/withStyles";

const styles = theme => ({
    toolbar: theme.mixins.toolbar,
});

const HeaderPlaceHolder = ({classes}) => (
    <div className={classes.toolbar}/>
);

export default withStyles(styles)(HeaderPlaceHolder);
