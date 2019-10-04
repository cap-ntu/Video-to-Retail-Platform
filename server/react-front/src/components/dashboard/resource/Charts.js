import React from "react";
import PropTypes from "prop-types";
import withStyles from "@material-ui/core/styles/withStyles";
import {ResponsiveLine} from '@nivo/line';
import Typography from "@material-ui/core/Typography";

const styles = {
    root: {
        height: "20rem",
        margin: 'auto',
        marginBottom: '2rem',
        '@media (max-width: 840px)': {
            height: '16rem',
        },
        '@media (max-width: 600px)': {
            height: "10rem"
        }
    },
    responsiveLine: {
        border: 1
    }
};

const commonProps = {
    margin: {"top": 10, "bottom": 50, "left": 60},
    colors: `red_yellow_blue`,
    xScale: {"type": "linear", "min": "auto"},
    yScale: {"type": "linear", "stacked": false, "min": 0, "max": 1},
    enableArea: true,
    animate: false,
    isInteractive: false,
    enableDots: false,
};

class Chart extends React.Component {

    render() {
        const {classes, data, injection={}} = this.props;
        return (
            <Typography variant={"h5"} className={classes.root}>
                <ResponsiveLine
                    data={data}
                    {...commonProps}
                    {...injection}
                    className={classes.responsiveLine}
                />
            </Typography>
        )
    }
}

Chart.propTypes = {
    classes: PropTypes.object.isRequired,
    data: PropTypes.arrayOf(
        PropTypes.object.isRequired,
    ).isRequired,
    injection: PropTypes.object,
};

export default withStyles(styles)(Chart);
