import React from "react";
import * as PropTypes from "prop-types";
import SnackbarContent from "@material-ui/core/SnackbarContent";
import IconButton from "@material-ui/core/IconButton";
import Icon_ from "@material-ui/core/Icon";
import green from "@material-ui/core/colors/green";
import MuiSnackbar from "@material-ui/core/Snackbar";
import CloseIcon from "@material-ui/icons/CloseRounded";
import CheckIcon from "@material-ui/icons/CheckRounded";
import ErrorIcon from "@material-ui/icons/ErrorRounded";
import withStyles from "@material-ui/core/styles/withStyles";
import classNames from "classnames";

const styles = theme => ({
    success: {
        backgroundColor: green[600],
    },
    failure: {
        backgroundColor: theme.palette.error.dark,
    },
    icon: {
        fontSize: 20,
    },
    iconVariant: {
        opacity: 0.9,
        marginRight: theme.spacing.unit,
    },
    message: {
        display: 'flex',
        alignItems: 'center',
    },
});

const validStates = ["SUCCESS", "FAILURE"];

const variantIcon = {
    success: CheckIcon,
    failure: ErrorIcon,
};


class Snackbar extends React.PureComponent {

    state = {
        open: false,
    };

    componentWillUpdate(nextProps, nextState, nextContext) {
        if (nextProps.state.state !== this.props.state.state &&
            validStates.includes(nextProps.state.state))
            this.setState({open: true});
    }

    handleClose = () => {
        this.setState({open: false});
    };

    render() {
        const {classes, className, state, message, ...other} = this.props;
        const {open} = this.state;

        const Icon = validStates.includes(state.state) ? variantIcon[state.state.toLowerCase()] : Icon_;

        if (!message["failure"])
            message["failure"] = `Some errors happen: ${state.reason}`;

        return (
            <MuiSnackbar
                open={open}
                anchorOrigin={{
                    vertical: 'bottom',
                    horizontal: 'left',
                }}
                autoHideDuration={6000}
                onClose={this.handleClose}
            >
                <SnackbarContent
                    className={classNames(classes[state.state.toLowerCase()], className)}
                    aria-describedby="client-snackbar"
                    message={
                        <span id="client-snackbar" className={classes.message}>
                            <Icon className={classNames(classes.icon, classes.iconVariant)}/>
                            {message[state.state.toLowerCase()]}
                        </span>}
                    action={[
                        <IconButton
                            key="close"
                            aria-label="Close"
                            color="inherit"
                            className={classes.close}
                            onClick={this.handleClose}
                        >
                            <CloseIcon className={classes.icon}/>
                        </IconButton>,
                    ]}
                    {...other}
                />
            </MuiSnackbar>
        );
    }
}

Snackbar.propTypes = {
    classes: PropTypes.object.isRequired,
    className: PropTypes.object,
    state: PropTypes.shape({
        state: PropTypes.string.isRequired,
        reason: PropTypes.string.isRequired,
    }).isRequired,
    message: PropTypes.shape({
        success: PropTypes.oneOfType([PropTypes.node, PropTypes.string]),
        failure: PropTypes.oneOfType([PropTypes.node, PropTypes.string]),
    }),
};

export default withStyles(styles)(Snackbar);
