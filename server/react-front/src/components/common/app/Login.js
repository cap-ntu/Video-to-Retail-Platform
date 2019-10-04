import React from "react";
import * as PropTypes from "prop-types";
import Button from "../Button";
import {deleteCookie, getCookie} from "../../../utils/utils";
import IconButton from "@material-ui/core/IconButton";
import PersonIcon from "@material-ui/icons/PersonRounded";
import Popper from "@material-ui/core/Popper";
import Paper from "@material-ui/core/Paper";
import ClickAwayListener from "@material-ui/core/ClickAwayListener";
import MenuList from "@material-ui/core/MenuList";
import MenuItem from "@material-ui/core/MenuItem";
import Divider from "@material-ui/core/Divider";
import Grow from "@material-ui/core/Grow";
import {USER_logout} from "../../../redux/actions/user/get";
import {connect} from "react-redux";

const mapStateToProps = () => ({});
const mapDispatchToProps = dispatch => ({
    handleLogout: successCallback => dispatch(USER_logout(successCallback))
});

class Login extends React.Component {
    state = {
        open: false,
    };

    anchorEl = null;

    handleToggle = () => {
        this.setState(state => ({open: !state.open}));
    };

    handleClose = event => {
        if (this.anchorEl.contains(event.target)) {
            return;
        }
        this.setState({open: false});
    };

    handleLogout = () => {
        const cleanUp = () => {
            this.setState({open: false}, () => window.location.reload());
            deleteCookie("csrftoken");
        };

        this.props.handleLogout(cleanUp);
    };

    render() {
        const {} = this.props;
        const {open} = this.state;
        return (
            !getCookie("csrftoken") ?
                <Button color="inherit">Login</Button> :
                <div>
                    <IconButton
                        color="inherit"
                        buttonRef={node => {
                            this.anchorEl = node;
                        }}
                        aria-haspopup="true"
                        onClick={this.handleToggle}
                    >
                        <PersonIcon/>
                    </IconButton>
                    <Popper open={open} anchorEl={this.anchorEl} transition disablePortal>
                        {({TransitionProps}) => (
                            <Grow {...TransitionProps}>
                                <Paper>
                                    <ClickAwayListener onClickAway={this.handleClose}>
                                        <MenuList>
                                            <MenuItem dense>Welcome</MenuItem>
                                            <Divider variant="middle"/>
                                            <MenuItem dense onClick={this.handleLogout}>Logout</MenuItem>
                                        </MenuList>
                                    </ClickAwayListener>
                                </Paper>
                            </Grow>
                        )}
                    </Popper>
                </div>
        );
    }
}

Login.propTypes = {
    classes: PropTypes.object,
    handleLogout: PropTypes.func.isRequired,
};

export default connect(mapStateToProps, mapDispatchToProps)(Login);
