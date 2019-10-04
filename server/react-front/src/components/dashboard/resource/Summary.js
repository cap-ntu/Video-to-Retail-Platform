import React from 'react';
import PropTypes from 'prop-types';
import Divider from "@material-ui/core/Divider";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText/ListItemText";
import List from "@material-ui/core/List";
import withStyles from "@material-ui/core/styles/withStyles";
import Chart from "./Charts";
import Grid from "@material-ui/core/Grid/Grid";
import RightNavigation from "../../common/RightNavigation";

const drawerWidth = '15rem';
const drawerWidthSm = '8rem';

const styles = theme => ({
    drawer: {
        width: drawerWidth,
        flexShrink: 0,
        '@media (max-width: 600px)': {
            width: drawerWidthSm,
        },
    },
    drawerPaper: {
        width: drawerWidth,
        '@media (max-width: 600px)': {
            width: drawerWidthSm
        }
    },
    toolbar: theme.mixins.toolbar,
    chart: {
        height: '3rem',
        marginBottom: '1rem',
    }
});

const injection = {
    margin: {bottom: 0},
    axisLeft: null,
    axisBottom: null,
    enableGridX: false,
    enableGridY: false,
    yScale: {"type": "linear", "stacked": false, "min": 0, "max": 1},
};

const resources = {
    type: ["CPU", "Memory", "Disk", "Network", "GPU"],
    colors: ['#A00036', '#F46D43', '#FEE08B', '#A6D96A', '#1A9850', '#41B6C4'],
    max: [1, 16, "auto", "auto", "auto"],
};

const Summary = ({classes, overall}) => (
    <RightNavigation classes={{drawer: classes.drawer, drawerPaper: classes.drawerPaper}}>
        <Divider/>
        <List>
            {resources.type.map((text, index) => (
                <ListItem button key={text}>
                    <Grid container spacing={16}>
                        <Grid item xs={6}>
                            <Chart data={overall[text] ? [overall[text]] : []}
                                   classes={{root: classes.chart}}
                                   injection={{...injection,
                                       colors: resources.colors[index],
                                       yScale: {...injection.yScale, max: resources.max[index]}}}/>
                        </Grid>
                        <Grid item xs={2}>
                            <ListItemText primary={text}/>
                        </Grid>
                    </Grid>
                </ListItem>
            ))}
        </List>
        <Divider/>
        <List>
            {['Summary', 'Power'].map((text) => (
                <ListItem button key={text}>
                    <ListItemText primary={text}/>
                </ListItem>
            ))}
        </List>
    </RightNavigation>
);

Summary.propTypes = {
    overall: PropTypes.object.isRequired,
};

export default withStyles(styles)(Summary);
