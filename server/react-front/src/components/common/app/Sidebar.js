import React from 'react';
import Drawer from "@material-ui/core/Drawer/Drawer";
import CssBaseline from "@material-ui/core/CssBaseline/CssBaseline";
import List from "@material-ui/core/List/List";
import ListItem from "@material-ui/core/ListItem/ListItem";
import ListItemText from "@material-ui/core/ListItemText/ListItemText";
import Divider from "@material-ui/core/Divider/Divider";
import PropTypes from "prop-types";
import {Link} from "react-router-dom";
import withStyles from "@material-ui/core/styles/withStyles";
import HeaderPlaceHolder from "../HeaderPlaceholder";

const styles = theme => ({
    root: {
        flexShrink: 0,
    },
    titleRoot: {
        paddingTop: 0,
        paddingBottom: 0,
    },
    drawerPaper: {
        backgroundColor: theme.palette.grey[50],
    },
    toolbar: theme.mixins.toolbar,
});

const Sidebar = ({classes, sidebarList, currentPath}) => (
    <Drawer variant="permanent"
            anchor="left"
            className={classes.root}
            classes={{paper: classes.drawerPaper}}>
        <HeaderPlaceHolder/>
        <CssBaseline/>
        <List>
            {sidebarList.map((item) => {
                const selected = item.url === currentPath;
                const color = selected ? "primary" : null;

                return (
                    <ListItem key={item.name}
                              component={props => <Link to={item.url} {...props}/>}
                              selected={selected}
                              disabled={item.disabled}
                              button>
                        {item.component ? React.cloneElement(item.component, {color: color || "action"}) : null}
                        <ListItemText primary={item.name}
                                      primaryTypographyProps={{
                                          variant: "subtitle2",
                                          noWrap: true,
                                          color: color || "default",
                                      }}/>
                    </ListItem>)
            })}
        </List>
        <Divider/>
        <List>
            {['Product Retrieval', 'Training Tracking', 'System Optimization'].map((text) => (
                <ListItem button key={text} disabled>
                    <ListItemText primary={text}
                                  primaryTypographyProps={{
                                      variant: "subtitle2",
                                      noWrap: true,
                                  }}/>
                </ListItem>
            ))}
        </List>
    </Drawer>
);

Sidebar.propTypes = {
    classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(Sidebar);
