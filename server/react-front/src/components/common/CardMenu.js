import React from 'react';
import PropTypes from 'prop-types';
import IconButton from "@material-ui/core/IconButton/IconButton";
import ClickAwayListener from "@material-ui/core/ClickAwayListener/ClickAwayListener";
import withStyles from "@material-ui/core/styles/withStyles";
import MenuList from "@material-ui/core/MenuList";
import MenuItem from "@material-ui/core/MenuItem";
import Paper from "@material-ui/core/Paper";
import Popper from "@material-ui/core/Popper/Popper";
import Grow from "@material-ui/core/Grow/Grow";
import MoreVertIcon from "@material-ui/icons/MoreVert";
import classNames from "classnames";
import Typography from "@material-ui/core/es/Typography/Typography";

const styles = {
    root: {},
    button: {},
    buttonMenuOpen: {},
};

class CardMenu extends React.Component {
    state = {
        open: false,
    };

    handleToggle = () => {
        this.setState(state => ({open: !state.open}));
    };

    handleClose = e => {
        if (this.anchorEl.contains(e.target)) {
            return;
        }

        this.setState({open: false});
    };

    render() {

        function mapFontSizeToVariant(fontSize) {
            switch (fontSize) {
                case "small":
                    return "body2";
                case "inherit":
                    return "inherit";
                case "large":
                    return "subtitle1";
                default:
                    return "default";
            }
        }

        const {classes, menuItems = [], fontSize = "default", disable = menuItems.length === 0} = this.props;
        const {open} = this.state;
        const dense = fontSize === "small";

        return (
            <div className={classes.root}>
                <IconButton className={classNames(classes.button, {[classes.buttonMenuOpen]: open})}
                            buttonRef={node => {
                                this.anchorEl = node;
                            }}
                            aria-owns={open ? 'menu-list-grow' : undefined}
                            aria-haspopup="true"
                            onClick={this.handleToggle}
                            disabled={disable || false}>
                    <MoreVertIcon fontSize={fontSize}/>
                </IconButton>
                {disable ?
                    null :
                    <Popper open={open} anchorEl={this.anchorEl} transition disablePortal style={{zIndex: 99}}>
                        {({TransitionProps, placement}) => (
                            <Grow
                                {...TransitionProps}
                                id="menu-list-grow"
                                style={{transformOrigin: placement === 'bottom' ? 'center top' : 'center bottom'}}>
                                <Paper>
                                    <ClickAwayListener onClickAway={this.handleClose}>
                                        <MenuList dense={dense}>
                                            {menuItems.map(
                                                (item, key) =>
                                                    <MenuItem dense={dense} key={key}
                                                              onClick={item.action}
                                                              component={item.url ? props => <a
                                                                  href={item.url} {...props}/> : null}>
                                                        {
                                                            React.isValidElement(item.id) ?
                                                                <Typography
                                                                    component="div"
                                                                    variant={mapFontSizeToVariant(fontSize)}>{item.id}
                                                                </Typography> :
                                                                item.id
                                                        }
                                                    </MenuItem>
                                            )}
                                        </MenuList>
                                    </ClickAwayListener>
                                </Paper>
                            </Grow>
                        )}
                    </Popper>
                }
            </div>
        );
    }
}

CardMenu.propTypes = {
    classes: PropTypes.object.isRequired,
    menuItems: PropTypes.arrayOf(
        PropTypes.shape({
            id: PropTypes.oneOfType([PropTypes.string, PropTypes.node]).isRequired,
            action: PropTypes.func,
            url: PropTypes.string,
        }).isRequired
    ),
    disable: PropTypes.bool,
    fontSize: PropTypes.oneOf(["small", "default", "inherit", "large"]),
};

export default withStyles(styles)(CardMenu);
