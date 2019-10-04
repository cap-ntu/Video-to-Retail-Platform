import React from 'react';
import PropTypes from 'prop-types';
import Card from '@material-ui/core/Card';
import CardContent from "@material-ui/core/CardContent";
import Typography from '@material-ui/core/Typography'
import Icon from "@material-ui/core/Icon/Icon";
import CardHeader from "@material-ui/core/CardHeader";
import Button from "@material-ui/core/Button";
import BuildIcon from '@material-ui/icons/Build';
import withStyles from "@material-ui/core/styles/withStyles";

const styles = {
    root: {
        width: '48rem',
        height: '24rem',
        margin: 'auto',
        marginBottom: '2rem',
    },
};

const DemoCard = ({classes}) =>
        (
            <Card className={classes.root}>
                <CardHeader
                    avatar={
                        <Icon>
                            <BuildIcon color={'primary'}/>
                        </Icon>
                    }
                    title={'Usage'}
                />
                <CardContent>
                    <Typography paragraph>
                        This website offers a online detection system for you to test your video and get the detection
                        results right at the points.
                    </Typography>
                    <Typography align={'left'} gutterBottom>
                        <Button color={'default'} href={'/demo'}>
                            Have a try
                        </Button>
                    </Typography>
                </CardContent>
            </Card>
        );

DemoCard.propTypes = {
    classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(DemoCard);
