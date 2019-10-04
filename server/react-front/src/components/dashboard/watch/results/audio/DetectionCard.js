import React from "react";
import AppBar from "@material-ui/core/AppBar";
import Typography from "@material-ui/core/Typography";
import Paper from "@material-ui/core/Paper";
import Toolbar from "@material-ui/core/Toolbar";
import HeaderPlaceHolder from "../../../../common/HeaderPlaceholder"
import DetectionTable from "../common/DetectionTable";
import withStyles from "@material-ui/core/styles/withStyles";

const styles = (theme) => ({
    root: {
        width: "100%",
        height: 420,
        display: "flex",
        flexDirection: "column",
        overflowY: "scroll",
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
});

const DetectionCard = ({classes, result, frame}) => (
    <div className={classes.root}>
        <Paper className={classes.paper} elevation={0} title="Audio result">
            <AppBar position="absolute" color="default" style={{backgroundColor: "white"}}>
                <Toolbar className={classes.toolbar}>
                    <Typography style={{flexGrow: 1}} variant="h5" component="h4">
                        Audio Result
                    </Typography>
                </Toolbar>
            </AppBar>
            <HeaderPlaceHolder/>
            <DetectionTable rows={result[frame]}/>
        </Paper>
    </div>
);

export default withStyles(styles)(DetectionCard);
