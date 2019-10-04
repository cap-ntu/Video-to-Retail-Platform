import React from "react";
import PropTypes from "prop-types";
import Paper from "@material-ui/core/Paper";
import classNames from "classnames";
import withStyles from "@material-ui/core/styles/withStyles";

const avatarSize = {
    small: 96,
    medium: 128,
    large: 224,
};

const styles = theme => ({
    root: {
        backgroundColor: theme.palette.grey[500],
        borderRadius: theme.shape.avatarBorderRadius,
        backgroundSize: "cover"
    },
    smallProps: {
        height: avatarSize.small,
        width: avatarSize.small,
    },
    mediumProps: {
        height: avatarSize.medium,
        width: avatarSize.medium,
    },
    largeProps: {
        height: avatarSize.large,
        width: avatarSize.large,
    }
});

const RoundCornerAvatar = ({classes, src, size = "medium"}) => (
    <Paper className={classNames(classes.root, classes[`${size}Props`])} component={'img'}
           src={src} elevation={1}/>
);

RoundCornerAvatar.propTypes = {
    classes: PropTypes.object.isRequired,
    src: PropTypes.string.isRequired,
    size: PropTypes.oneOf(["small", "medium", "large"])
};

export default withStyles(styles)(RoundCornerAvatar);
