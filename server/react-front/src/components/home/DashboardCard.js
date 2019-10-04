import Card from "@material-ui/core/Card/Card";
import CardHeader from "@material-ui/core/CardHeader/CardHeader";
import Icon from "@material-ui/core/Icon/Icon";
import DashboardRoundedIcon from "@material-ui/icons/DashboardRounded";
import CardContent from "@material-ui/core/CardContent/CardContent";
import Typography from "@material-ui/core/Typography/Typography";
import Button from "@material-ui/core/Button/Button";
import PropTypes from "prop-types";
import React from "react";
import withStyles from "@material-ui/core/styles/withStyles";

const styles = {
    root: {
        width: '48rem',
        height: '24rem',
        margin: 'auto',
        marginBottom: '2rem',
    },
};

const DashboardCard = ({classes}) =>
        (
            <Card className={classes.root}>
                <CardHeader
                    avatar={
                        <Icon>
                            <DashboardRoundedIcon color={'primary'}/>
                        </Icon>
                    }
                    title={'Dashboard'}
                />
                <CardContent>
                    <Typography paragraph>
                        See how your server performs using our Hysia platform
                    </Typography>
                    <Typography align={'left'} gutterBottom>
                        <Button color={'default'} href={'/dashboard'}>
                            Simulate
                        </Button>
                    </Typography>
                </CardContent>
            </Card>
        );

DashboardCard.propTypes = {
    classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(DashboardCard);
