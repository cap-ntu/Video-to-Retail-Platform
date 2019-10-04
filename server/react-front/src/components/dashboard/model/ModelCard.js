import React from "react";
import Paper from "@material-ui/core/Paper";
import Grid from "@material-ui/core/Grid";
import Typography from "@material-ui/core/Typography";
import PropTypes from "prop-types";
import Button from "../../common/Button";
import withStyles from "@material-ui/core/styles/withStyles";
import Divider from "@material-ui/core/Divider";
import Link from "react-router-dom/es/Link";
import RoundCornerAvatar from "../../common/RoundCornerAvatar";

const styles = theme => ({
    root: {
        backgroundColor: "transparent",
        marginBottom: 2 * theme.spacing.unit,
        maxHeight: 210,
    },
    avatar: {
        backgroundColor: theme.palette.grey[500],
        borderRadius: theme.shape.avatarBorderRadius,
        height: 128,
        width: 128,
    },
    description: {
        width: 342,
        height: "100%",
    },
    divider: {
        marginTop: 16,
    }
});

const ModelCard = ({classes, id, name, cover}) => (
    <Paper className={classes.root} elevation={0} component={props => <Link {...props}/>}
           to={{pathname: './search', search: `id=${id}`}}>
        <Grid spacing={24} container>
            <Grid item>
                <RoundCornerAvatar
                    src={cover || `https://picsum.photos/128/?image=${Math.round(Math.random() * 200)}`}/>
            </Grid>
            <Grid item xs>
                <Grid className={classes.description} direction={'column'} spacing={16} container>
                    <Grid item xs>
                        <Typography variant={"h6"} gutterBottom>{name || "Title"}</Typography>
                        <Typography variant={"body2"} gutterBottom>Hysia Team</Typography>
                    </Grid>
                    <Grid item>
                        <Button variant={"contained"} color={"primary"} aria-label={"details"} size={"small"}>
                            Get
                        </Button>
                    </Grid>
                </Grid>
                <Divider className={classes.divider} variant={"middle"}/>
            </Grid>
        </Grid>
    </Paper>
);

ModelCard.propTypes = {
    classes: PropTypes.object.isRequired,
    id: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    cover: PropTypes.string,
};

export default withStyles(styles)(ModelCard);
