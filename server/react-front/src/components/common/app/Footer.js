import React from "react";
import Grid from "@material-ui/core/Grid";
import Typography from "@material-ui/core/Typography";
import withStyles from "@material-ui/core/styles/withStyles"
import Link from "../Link";

const styles = theme => {
    return ({
        root: {
            flexGrow: 1,
            position: "sticky",
            top: 0,
            paddingTop: 20,
            backgroundColor: theme.palette.grey[100],
            width: "100%",
            zIndex: 1250,
        },
        container: {
            width: "70%",
            maxWidth: 1280,
            margin: [["0px", "auto"]]
        },
    });
};

const links = [
    ["GitHub", "https://github.com/orgs/cap-ntu/teams/hysia"],
    ["Team", "/team"],
    ["Demo", "/dashboard/video/upload"]
];

const Footer = ({classes}) => (
    <footer className={classes.root}>
        <div className={classes.container}>
            <Typography component="div" variant="h5" gutterBottom>
                Quick Links
            </Typography>
            <Grid container spacing={8}>
                {links.map(([name, link], index) =>
                    <Grid item key={index} xs={6}>
                        <Link to={link} animation={false} paragraph>
                            <Typography variant="subtitle1" gutterBottom>{name}</Typography>
                        </Link>
                    </Grid>
                )}
            </Grid>
            <Typography>Hysia 1.0</Typography>
        </div>
    </footer>
);

export default withStyles(styles)(Footer);
