import React from "react";
import * as PropTypes from "prop-types";
import withStyles from "@material-ui/core/styles/withStyles";
import Divider from "@material-ui/core/es/Divider/Divider";
import Typography from "@material-ui/core/es/Typography/Typography";
import Grid from "@material-ui/core/Grid";
import ModelCard from "./ModelCard";
import Button from "../../common/Button";
import CardHeader from "@material-ui/core/CardHeader";
import Fade from "@material-ui/core/es/Fade/Fade";
import Link from "react-router-dom/es/Link";


const styles = theme => ({
    root: {
        width: '85%',
        margin: 'auto',
    },
    models: {
        padding: [[3 * theme.spacing.unit, 0]],
    },
    divider: {
        margin: [[theme.spacing.unit, 0]],
        "&:first-child": {
            opacity: 0,
        }
    }
});

class ModelManagement extends React.PureComponent {

    componentWillMount() {
        this.props.fetchModelList();
    }

    render() {
        const {classes, models} = this.props;

        return (
            <Fade in={true}>
                <div className={classes.root}>
                    <CardHeader action={<Button component={props => <Link {...props}/>}
                                                to={'./upload'}>Add</Button>}/>
                    <div>{
                        Object.keys(models).length > 0 ?
                            Object.keys(models).map(key => (
                                <React.Fragment key={key}>
                                    <Divider className={classes.divider}/>
                                    <div className={classes.models}>
                                        <Typography variant={"h5"} gutterBottom>
                                            {`${key.charAt(0).toUpperCase()}${key.slice(1).toLowerCase()}`}
                                        </Typography>
                                        <Grid spacing={32} container>{
                                            models[key].map(model =>
                                                <Grid key={model.id} item>
                                                    <ModelCard {...model}/>
                                                </Grid>)
                                        }
                                        </Grid>
                                    </div>
                                </React.Fragment>
                            )) :
                            <Typography align={"center"}>
                                You haven't install any model.
                            </Typography>
                    }
                    </div>
                </div>
            </Fade>);
    }
}

ModelManagement.propTypes = {
    classes: PropTypes.object.isRequired,
    models: PropTypes.shape({
        [PropTypes.oneOf(["OBJECT", "TEXT", "FACE", "SCENE"])]: PropTypes.arrayOf(
            PropTypes.shape({
                name: PropTypes.string.isRequired,
                id: PropTypes.string.isRequired,
            }).isRequired,
        ),
    }).isRequired,
    fetchModelList: PropTypes.func.isRequired,
};

export default withStyles(styles)(ModelManagement);
