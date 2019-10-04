import withStyles from "@material-ui/core/styles/withStyles";
import Grid from "@material-ui/core/Grid";
import React from "react";
import LibraryAddIcon from "@material-ui/icons/LibraryAddRounded";
import {styles} from "./PlayOverlay";

export default withStyles(styles)(({classes}) => (
    <Grid className={classes.overlay} style={{opacity: 1}} direction={"row"} justify={"center"} container>
        <Grid item style={{margin: "auto"}}>
            <LibraryAddIcon className={classes.overlayIcon}/>
            Upload a new video
        </Grid>
    </Grid>
));
