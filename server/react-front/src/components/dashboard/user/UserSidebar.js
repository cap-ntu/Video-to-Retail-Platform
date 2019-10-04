import React from "react";
import PropTypes from "prop-types";
import Drawer from "@material-ui/core/es/Drawer/Drawer";
import HeaderPlaceHolder from "../../common/HeaderPlaceholder";
import CssBaseline from "@material-ui/core/CssBaseline";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";
import ListItemAvatar from "@material-ui/core/es/ListItemAvatar/ListItemAvatar";
import withStyles from "@material-ui/core/styles/withStyles";
import Avatar from "@material-ui/core/es/Avatar/Avatar";
import CardHeader from "@material-ui/core/es/CardHeader/CardHeader";
import IconButton from "@material-ui/core/es/IconButton/IconButton";
import AddIcon from "@material-ui/icons/AddRounded";

const styles = theme => ({
    root: {
        width: 320,
        flexShrink: 0,
    },
    paper: {
        width: 320,
        left: 240,
        backgroundColor: theme.palette.grey[100],
    }
});

const UserSidebar = ({classes, nameList, currentUser, newUser, handleClick, handleNewUser}) => (
    <Drawer className={classes.root} classes={{paper: classes.paper}}
            variant="permanent" anchor="left" color="primary" elevation={0}>
        <HeaderPlaceHolder/>
        <CssBaseline/>
        <CardHeader action={
            <IconButton onClick={handleNewUser} disabled={newUser}>
                <AddIcon fontSize="small"/>
            </IconButton>}/>
        <List>
            {Object.keys(nameList).length === 0 ?
                <ListItem>
                    <ListItemText inset>
                        No user currently.
                    </ListItemText>
                </ListItem> :
                Object.keys(nameList).map(item => {
                    return (
                        <React.Fragment key={item}>
                            <ListItem>
                                <ListItemText primary={item}/>
                            </ListItem> {
                            nameList[item].map(name => <ListItem key={name} onClick={() => handleClick(name)}
                                                                 selected={name === currentUser} button>
                                <ListItemAvatar>
                                    <Avatar>{name.slice(0, 2).toUpperCase()}</Avatar>
                                </ListItemAvatar>
                                <ListItemText primary={name}
                                              primaryTypographyProps={{
                                                  variant: "subtitle2",
                                                  noWrap: true,
                                                  color: "default",
                                              }}/>
                            </ListItem>)
                        }
                        </React.Fragment>)
                })}
        </List>
    </Drawer>
);

UserSidebar.defaultProps = {
    currentUser: "",
};

UserSidebar.propTypes = {
    classes: PropTypes.object.isRequired,
    nameList: PropTypes.object.isRequired,
    currentUser: PropTypes.string.isRequired,
    handleClick: PropTypes.func.isRequired,
    handleNewUser: PropTypes.func.isRequired,
};

export default withStyles(styles)(UserSidebar)
