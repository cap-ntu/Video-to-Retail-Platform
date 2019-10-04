import React from "react";
import * as PropTypes from "prop-types";
import Cpu from "./Cpu";
import Summary from "./Summary";
import withStyles from "@material-ui/core/styles/withStyles";

const styles = {};

class ResourceManagement extends React.PureComponent {

    componentWillMount() {
        this.interval = setInterval(this.props.fetchStatistics, 1000);
    }

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    render() {
        const {data} = this.props;
        return (
            <React.Fragment>
                <Cpu data={data.CPU} overall={data.summary.CPU ? [data.summary.CPU] : []}/>
                <Summary overall={data.summary}/>
            </React.Fragment>
        );
    }
}

ResourceManagement.propTypes = {classes: PropTypes.object.isRequired};

export default withStyles(styles)(ResourceManagement);
