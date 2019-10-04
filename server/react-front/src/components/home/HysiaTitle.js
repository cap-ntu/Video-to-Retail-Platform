import React from 'react';
import * as PropTypes from 'prop-types';
import Typography from "@material-ui/core/Typography";
import {withStyles} from "@material-ui/core/styles";
import Button from "../common/Button";
import Link from "../common/Link";


const styles = {
    root: {
        maxWidth: 300,
        margin: "auto",
        padding: "48px 0px",
        '@media (min-width: 600px)': {
            paddingTop: 64,
            maxWidth: 500,
        },
    },
    logo: {
        margin: "auto",
        maxHeight: 382,
    },
    h4: {
        lineHeight: "2.5rem",
        fontWeight: 200,
    },
};

const HysiaTitle = ({classes}) => (
    <div className={classes.root}>
        <Typography className={classes.logo} variant="h1" component="img" align="center"
                    src={require('../../asset/hysia logo.png')} alt={'hysia-logo'}/>
        <Typography variant="h2" align="center" color="primary" gutterBottom>
            HYSIA
        </Typography>
        <Typography variant={'h4'} align={'center'} paragraph className={classes.h4}>
            a low-latency high-throughput
            video analysis system.
        </Typography>
        <Typography variant={'h3'} align={'center'}>
            <Link to='/dashboard/' animation={false}>
                <Button variant={'outlined'} color={'primary'} className={classes.button}>
                    Try our dashboard!
                </Button>
            </Link>
        </Typography>
    </div>
);

HysiaTitle.propTypes = {
    classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(HysiaTitle);
