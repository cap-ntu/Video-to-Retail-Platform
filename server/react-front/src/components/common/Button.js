import Button from "@material-ui/core/Button";
import React from "react";
import withStyles from "@material-ui/core/styles/withStyles";

const styles = theme => ({
    root: {
        borderRadius: theme.shape.buttonBorderRadius,
    },
});

export default withStyles(styles)(({classes, ...props}) => <Button
    classes={{root: classes.root,}} {...props}/>);
