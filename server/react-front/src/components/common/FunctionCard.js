import React from 'react';
import PropTypes from 'prop-types';
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import CardHeader from "@material-ui/core/CardHeader";
import classNames from "classnames";
import withStyles from "@material-ui/core/styles/withStyles"
import CardMenu from "./CardMenu";

const styles = {
    root: {
        boxShadow: 'inherit'
    },
};

const FunctionCard = ({classes, children, title, menuItems, className}) => (
    <Card className={classNames(className, classes.root)}>
        <CardHeader
            action={
                <CardMenu menuItems={menuItems} disable={menuItems == null}/>
            }
            title={title}/>
        <CardContent>
            {children}
        </CardContent>
    </Card>
);

FunctionCard.propTypes = {
    classes: PropTypes.object.isRequired,
    children: PropTypes.node,
    title: PropTypes.string,
    menuItems: PropTypes.arrayOf(
        PropTypes.shape({
            name: PropTypes.isRequired,
            url: PropTypes.isRequired,
        }).isRequired,
    ),
    className: PropTypes.string,
};

export default withStyles(styles)(FunctionCard);
