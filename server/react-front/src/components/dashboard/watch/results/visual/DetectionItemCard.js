import React from "react";
import * as PropTypes from "prop-types";
import Avatar from "@material-ui/core/es/Avatar/Avatar";
import Typography from "@material-ui/core/es/Typography/Typography";
import withStyles from "@material-ui/core/styles/withStyles";
import Grid from "@material-ui/core/Grid";
import Paper from "@material-ui/core/Paper";
import Grow from "@material-ui/core/es/Grow/Grow";
import Collapse from "@material-ui/core/es/Collapse/Collapse";

const styles = theme => ({
    root: {
        backgroundColor: theme.palette.grey[100],
        margin: `${theme.spacing.unit}px auto`,
        padding: 2 * theme.spacing.unit,
    }
});

const DetectionItemCard = ({classes, on, id, cat, model, type, detectionSrc, conf}) => (
    <Collapse in={on}>
        <Grow in={on}>
            <Paper className={classes.root} elevation={0}>
                <Grid spacing={16} justify="center" wrap="nowrap" container>
                    <Grid item>
                        <Avatar src={detectionSrc || `https://api.adorable.io/avatars/285/${cat}`}/>
                    </Grid>
                    <Grid item xs>
                        <Typography variant="subtitle1">
                            {cat}
                        </Typography>
                        <Typography variant="body2">
                            {`Detected by ${model} model with a confidence of ${conf * 100}%`}
                        </Typography>
                    </Grid>
                </Grid>
            </Paper>
        </Grow>
    </Collapse>
);

DetectionItemCard.defaultProps = {
    model: "some model",
    cat: "name",
    type: "OBJECT",
    conf: 1,
};

DetectionItemCard.propTypes = {
    classes: PropTypes.object.isRequired,
    on: PropTypes.bool,
    id: PropTypes.string.isRequired,
    model: PropTypes.string.isRequired,
    type: PropTypes.oneOf(["FACE", "OBJECT", "TEXT", "SCENE"]),
    cat: PropTypes.string,
    detectionSrc: PropTypes.string,
    conf: PropTypes.number,
};

export default withStyles(styles)(DetectionItemCard)
