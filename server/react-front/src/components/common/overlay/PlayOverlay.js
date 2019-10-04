import withStyles from "@material-ui/core/styles/withStyles";
import Grid from "@material-ui/core/Grid";
import PlayArrowIcon from "@material-ui/icons/PlayArrowRounded"
import React from "react";

export const styles = (theme) => ({
    overlay: {
        position: "absolute",
        height: "100%",
        opacity: 0,
        transition: theme.transitions.create("opacity", {
            duration: theme.transitions.duration.shortest,
        }),
        "&:hover": {
            opacity: 1,
        }
    },
    overlayIcon: {
        marginRight: theme.spacing.unit,
    },
});

export default withStyles(styles)(({classes}) => (
    <Grid className={classes.overlay} direction={"row"} justify={"center"} container>
        <Grid item style={{margin: "auto"}}>
            <PlayArrowIcon/>
        </Grid>
    </Grid>
));
