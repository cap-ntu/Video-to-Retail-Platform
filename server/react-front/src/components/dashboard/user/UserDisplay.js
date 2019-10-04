import React from "react";
import * as PropTypes from "prop-types";
import Grid from "@material-ui/core/Grid";
import Avatar from "@material-ui/core/Avatar";
import Typography from "@material-ui/core/Typography";
import Divider from "@material-ui/core/Divider";
import withStyles from "@material-ui/core/styles/withStyles";
import Button from "../../common/Button";
import CardContent from "@material-ui/core/CardContent";

const styles = theme => ({
    header: {},
    section: {},
    sectionInner: {},
    cardGridBg: {
        padding: theme.spacing.unit,
    },
    cardGridSm: {
        padding: 0.5 * theme.spacing.unit,
    },
    avatar: {
        width: 60,
        height: 60,
    }
});

function UserDisplay({classes, user, handleEdit}) {

    return (
        <React.Fragment>
            <div className={classes.header}>
                <div style={{flexGrow: 1}}/>
                <Button color={"primary"} onClick={handleEdit}>
                    Edit
                </Button>
            </div>
            <CardContent>
                <Grid className={classes.cardGridBg} justify={"center"} container>
                    <Avatar className={classes.avatar}>{user.username.slice(0, 2).toUpperCase()}</Avatar>
                </Grid>
                {
                    user.name ?
                        <Grid className={classes.cardGridSm} justify={"center"} container>
                            <Typography variant={"h6"}>{user.name}</Typography>
                        </Grid> : null
                }
                <Grid className={classes.cardGridSm} justify={"center"} container>
                    <Typography variant={user.name ? "subtitle1" : "h6"}>{user.username}</Typography>
                </Grid>
                <Divider/>
                <Grid className={classes.cardGridSm} spacing={8} container> {
                    user.email ?
                        <Grid xs item>
                            <Typography color={"primary"}>Email</Typography>
                            <Typography component={'a'}
                                        href={`mailto:${user.email}`}>{user.email}</Typography>
                        </Grid> : null
                }
                </Grid>
                <Divider/>
                <Grid className={classes.cardGridSm} spacing={8} container> {
                    user.groups.length !== 0 ?
                        <Grid item>
                            <Typography color={"primary"}>Group</Typography>
                            <Typography>{user.groups}</Typography>
                        </Grid> : null
                }
                </Grid>
                <Divider/>
                <Grid className={classes.cardGridSm} spacing={8} container>
                    <Grid xs item>
                        <Typography color={"primary"}>Account Type</Typography>
                        <Typography>{user.accountType}</Typography>
                    </Grid>
                    <Grid xs item>
                        <Typography color={"primary"}>Domain</Typography>
                        <Typography>{user.domain}</Typography>
                    </Grid>
                    <Grid xs item>
                        <Typography color={"primary"}>Status</Typography>
                        <Typography>{user.status}</Typography>
                    </Grid>
                </Grid>
            </CardContent>
        </React.Fragment>
    )
}

UserDisplay.propTypes = {
    classes: PropTypes.object.isRequired,
    user: PropTypes.shape({
        username: PropTypes.string.isRequired,
        firstName: PropTypes.string,
        lastName: PropTypes.string,
        email: PropTypes.string,
        accountType: PropTypes.oneOf(["Administrator", "User"]),
        domain: PropTypes.oneOf(["Staff", "Public"]),
        status: PropTypes.oneOf(["Activated", "Not Activated"]),
    }).isRequired,
    handleEdit: PropTypes.func.isRequired,
};

export default withStyles(styles)(UserDisplay);
