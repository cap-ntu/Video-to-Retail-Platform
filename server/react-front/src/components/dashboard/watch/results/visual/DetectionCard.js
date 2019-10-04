import React from "react";
import * as PropTypes from "prop-types";
import AppBar from "@material-ui/core/AppBar/AppBar";
import CubeIcon from "mdi-material-ui/Cube";
import FaceIcon from "@material-ui/icons/FaceRounded";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import LocalMoviesIcon from "@material-ui/icons/LocalMoviesRounded";
import Paper from "@material-ui/core/Paper";
import Switch from "@material-ui/core/Switch";
import Tab from "@material-ui/core/Tab";
import Tabs from "@material-ui/core/Tabs";
import TextFormatIcon from "@material-ui/icons/TextFormatRounded";
import Typography from "@material-ui/core/Typography";
import withStyles from "@material-ui/core/styles/withStyles";
import {IconButton, Toolbar} from "@material-ui/core";
import HeaderPlaceHolder from "../../../../common/HeaderPlaceholder";
import ExportIcon from "mdi-material-ui/ExportVariant";
import DetectionTable from "../common/DetectionTable";

const styles = (theme) => ({
    root: {
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        overflow: "scroll",
    },
    paper: {
        flexGrow: 1,
        padding: 2 * theme.spacing.unit,
        height: "100%",
        marginBottom: 2.5 * theme.spacing.unit,
    },
    toolbar: {
        paddingTop: theme.spacing.unit,
        paddingBottom: theme.spacing.unit,
    },
    appBar: {
        top: "auto",
        bottom: 0,
    },
    smallTab: {
        minWidth: 50,
    },
});

class DetectionCard extends React.PureComponent {

    state = {
        tab: "object",
    };

    handleTabChange = (event, value) => {
        this.setState({tab: value});
    };

    render() {
        const {classes, result, frame} = this.props;
        const {tab} = this.state;

        const card = {};
        Object.keys(result).forEach(key => card[key] = result[key][frame] || []);
        card.overall = Object.values(card).reduce((x, y) => x.concat(y), []);

        return <React.Fragment>
            <div className={classes.root}>
                <Paper className={classes.paper} elevation={0} title="Analytic result">
                    <AppBar position="absolute" color="default" style={{backgroundColor: "white"}}>
                        <Toolbar className={classes.toolbar}>
                            <Typography style={{flexGrow: 1}} variant="h5" component="h4">
                                Analytic Result
                            </Typography>
                            <FormControlLabel style={{flexShrink: 0}}
                                              control={
                                                  <Switch
                                                      checked={tab === "overall"}
                                                      onChange={e => this.handleTabChange(e, "overall")}
                                                      value="overall"
                                                  />}
                                              label="Show Overall"/>
                            <IconButton>
                                <ExportIcon/>
                            </IconButton>
                        </Toolbar>
                    </AppBar>
                    <HeaderPlaceHolder/>
                    {
                        card[tab] ?
                            <DetectionTable rows={card[tab]}/> :
                            <Typography align="center" variant="subtitle1" color="textSecondary">
                                Not processed by {tab} model
                            </Typography>
                    }
                </Paper>
                <HeaderPlaceHolder/>
            </div>
            <AppBar className={classes.appBar} position="absolute" color="default">
                <Tabs value={tab} onChange={this.handleTabChange}
                      indicatorColor="primary"
                      textColor="primary"
                      variant="fullWidth">
                    <Tab className={classes.smallTab} label="Object" icon={<CubeIcon/>} value="object"/>
                    <Tab className={classes.smallTab} label="Face" icon={<FaceIcon/>} value="face"/>
                    <Tab className={classes.smallTab} label="Text" icon={<TextFormatIcon/>} value="text"/>
                    <Tab className={classes.smallTab} label="Scene" icon={<LocalMoviesIcon/>} value="scene"/>
                    <Tab style={{display: "none"}} value={"overall"}/>
                </Tabs>
            </AppBar>
        </React.Fragment>;
    }
}

DetectionCard.defaultProps = {
    result: {
        object: [],
        face: [],
        text: [],
        scene: [],
    },
};

DetectionCard.propTypes = {
    classes: PropTypes.object.isRequired,
    result: PropTypes.shape({
        object: PropTypes.array,
        face: PropTypes.array,
        text: PropTypes.array,
        scene: PropTypes.array,
    }),
    frame: PropTypes.number.isRequired,
};

export default withStyles(styles)(DetectionCard)
