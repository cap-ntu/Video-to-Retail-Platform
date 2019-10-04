import React from "react";
import PropTypes from "prop-types";
import Typography from "@material-ui/core/Typography";
import withStyles from "@material-ui/core/styles/withStyles";
import Grid from "@material-ui/core/Grid";
import classNames from "classnames";

const styles = theme => ({
    icon: {
        display: "block"
    },
    leftIcon: {
        marginRight: theme.spacing.unit,
    },
    rightIcon: {
        marginLeft: theme.spacing.unit,
    },
});

const IconTypography = ({classes, iconPosition = "right", text, textProps, icon: Icon, iconProps}) => {

    const typography = <Grid item><Typography component={'p'} {...textProps}>{text}</Typography></Grid>;
    const icon = <Grid item><Icon className={classNames(classes.icon, classes[`${iconPosition}Icon`])} {...iconProps}/></Grid>;

    return (<Grid alignItems="center" justify="flex-start" container>
        {iconPosition === "left" ?
            <React.Fragment>{icon}{typography}</React.Fragment> :
            <React.Fragment>{typography}{icon}</React.Fragment>}
    </Grid>)
};

IconTypography.propTypes = {
    classes: PropTypes.object.isRequired,
    iconPosition: PropTypes.oneOf(["left", "right"]),
    text: PropTypes.string,
    textProps: PropTypes.object,
    icon: PropTypes.oneOfType([PropTypes.node, PropTypes.func]).isRequired,
    iconProps: PropTypes.object,
};

export default withStyles(styles)(IconTypography);
