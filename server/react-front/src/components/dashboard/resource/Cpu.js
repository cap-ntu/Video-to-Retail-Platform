import React from 'react';
import * as PropTypes from 'prop-types';
import Charts from "./Charts";
import Grid from "@material-ui/core/Grid/Grid";
import withStyles from "@material-ui/core/styles/withStyles";
import Typography from "@material-ui/core/Typography/Typography";
import Button from "@material-ui/core/Button/Button";
import Menu from "@material-ui/core/Menu/Menu";
import MenuItem from "@material-ui/core/MenuItem/MenuItem";

const styles = {
    root: {
        flexGrow: 1,
        minWidth: 0,
    },
    overall: {
        height: "48rem",
        '@media (max-width: 840px)': {
            height: '36rem',
        },
        '@media (max-width: 600px)': {
            height: "24rem"
        }
    }
};

const injection = {
    axisLeft: {
        orient: "bottom",
        tickSize: 5,
        tickPadding: 5,
        legendOffset: -40,
        legendPosition: "middle",
        legend: "CPU Utilization",
    },
    axisBottom: {
        orient: "left",
        tickSize: 0,
        legendOffset: 36,
        legendPosition: "middle",
        legend: "time",
    },
};

class Cpu extends React.Component {

    state = {
        anchorEl: null,
        isOverall: true,
        utilization: 0,
    };

    componentDidMount() {
        this._interval = setInterval(() => {
            const {overall} = this.props;
            this.setState({
                utilization:
                    overall.length === 0 ? 0 : Math.round(overall[0].data[overall[0].data.length - 1].y * 100)
            })
        }, 1000);
    }

    componentWillUnmount() {
        clearInterval(this._interval);
    }

    handleClick = event => {
        this.setState({anchorEl: event.currentTarget});
    };

    handleClose = (isOverall) => {
        this.setState({anchorEl: null, isOverall: isOverall});
    };

    render() {
        const {classes, data, overall} = this.props;
        const {anchorEl, isOverall, utilization} = this.state;

        if (window.innerWidth < 840)
            injection.legends = [];
        else
            delete injection.legends;

        return (
            <div className={classes.root}>
                <Grid container alignItems={'stretch'} spacing={24}>
                    {isOverall ?
                        <Grid item xs={12}>
                            <Charts classes={{root: classes.overall}}
                                    data={overall}
                                    injection={injection}/>
                        </Grid> :
                        data.map((cpu, index) =>
                            <Grid key={index} item sm={12} md={8} lg={3} xl={2}>
                                <Charts
                                    injection={injection}
                                    data={[cpu]}/>
                            </Grid>
                        )
                    }
                </Grid>
                <Grid>
                    <Typography>Utilization: {utilization}%</Typography>
                </Grid>
                <div>
                    <Button
                        aria-owns={anchorEl ? "display-mode-menu" : undefined}
                        aria-haspopup={"true"}
                        onClick={this.handleClick}
                    >
                        {isOverall ? "Overall" : "Logical Processors"}
                    </Button>
                    <Menu
                        id={"display-mode-menu"}
                        anchorEl={anchorEl}
                        open={Boolean(anchorEl)}
                        onClose={this.handleClose}
                    >
                        <MenuItem onClick={() => this.handleClose(true)}>Overall</MenuItem>
                        <MenuItem onClick={() => this.handleClose(false)}>Logical Processors</MenuItem>
                    </Menu>
                </div>
            </div>
        )
    }
}

Cpu.propTypes = {
    classes: PropTypes.object.isRequired,
    data: PropTypes.arrayOf(
        PropTypes.object.isRequired,
    ).isRequired,
    overall: PropTypes.arrayOf(
        PropTypes.object.isRequired,
    ).isRequired,
};

export default withStyles(styles)(Cpu);
